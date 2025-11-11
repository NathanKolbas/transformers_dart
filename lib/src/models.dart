// ignore_for_file: non_constant_identifier_names

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:transformers/extensions/list_extensions.dart';
import 'package:transformers/src/backends/onnx.dart' hide Tensor;
import 'package:transformers/src/configs.dart';
import 'package:transformers/src/generation/configuration_utils.dart';
import 'package:transformers/src/generation/logits_process.dart';
import 'package:transformers/src/generation/logits_sampler.dart';
import 'package:transformers/src/generation/stopping_criteria.dart';
import 'package:transformers/src/utils/constants.dart';
import 'package:transformers/src/utils/core.dart';
import 'package:transformers/src/utils/devices.dart';
import 'package:transformers/src/utils/dtypes.dart';
import 'package:transformers/src/utils/hub.dart';
import 'package:transformers/src/utils/tensor.dart';

//////////////////////////////////////////////////
// Model types: used internally
enum MODEL_TYPES {
  EncoderOnly(0),
  EncoderDecoder(1),
  Seq2Seq(2),
  Vision2Seq(3),
  DecoderOnly(4),
  MaskGeneration(5),
  ImageTextToText(6),
  Musicgen(7),
  MultiModality(8),
  Phi3V(9),
  AudioTextToText(10),
  AutoEncoder(11);

  final int value;

  const MODEL_TYPES(this.value);
}
//////////////////////////////////////////////////


//////////////////////////////////////////////////
// Helper functions

// NOTE: These will be populated fully later
final Map<String, MODEL_TYPES> MODEL_TYPE_MAPPING = {};
final Map<String, Type> MODEL_NAME_TO_CLASS_MAPPING = {};
final Map<Type, String> MODEL_CLASS_TO_NAME_MAPPING = {};

/// Constructs an InferenceSession using a model file located at the specified path.
/// @param {string} pretrained_model_name_or_path The path to the directory containing the model file.
/// @param {string} fileName The name of the model file.
/// @param {import('./utils/hub.js').PretrainedModelOptions} options Additional options for loading the model.
/// @returns {Promise<{buffer_or_path: Uint8Array|string, session_options: Object, session_config: Object}>} A Promise that resolves to the data needed to create an InferenceSession object.
/// @private
Future<(String, OrtSessionOptions, Map<String, dynamic>)> getSession(
  String pretrained_model_name_or_path,
  String fileName,
  PretrainedModelOptions options,
) async {
  TransformersJSConfig custom_config = options.config?.transformersJsConfig ?? TransformersJSConfig();

  dynamic device = options.device ?? custom_config.device;
  if (device != null && device is! DeviceType) {
    device = device as Map<String, DeviceType>;

    if (device.containsKey(fileName)) {
      device = device[fileName];
    } else {
      // console.warn
      print('device not specified for "$fileName". Using the default device.');
      device = null;
    }
  }

  // At this point we know device has to be a DeviceType?
  device = device as DeviceType?;

  // If the device is not specified, we use the default (supported) execution providers.
  final DeviceType selectedDevice = device ?? DeviceType.cpu;

  final executionProviders = await deviceToExecutionProviders(selectedDevice);

  // Update custom config with the selected device's config, if it exists
  final device_config = custom_config.device_config ?? {};
  if (device_config.containsKey(selectedDevice)) {
    custom_config = TransformersJSConfig.fromJson({
      ...custom_config.toJson(),
      ...?device_config[selectedDevice]?.toJson(),
    });
  }

  // If options.dtype is specified, we use it to choose the suffix for the model file.
  // Otherwise, we use the default dtype for the device.
  dynamic dtype = options.dtype ?? custom_config.dtype;
  if (dtype is! DataType) {
    dtype = dtype as Map<String, DataType>?;

    if (dtype != null && dtype.containsKey(fileName)) {
      dtype = dtype[fileName];
    } else {
      dtype = DEFAULT_DEVICE_DTYPE_MAPPING[selectedDevice] ?? DataType.fp32;
      // console.warn
      print('dtype not specified for "$fileName". Using the default dtype ($dtype) for this device (${selectedDevice.value}).');
    }
  }

  // At this point we know dtype has to be a DataType
  dtype = dtype as DataType;

  if (dtype == DataType.auto) {
    // Try to choose the auto dtype based on the custom config
    dynamic config_dtype = custom_config.dtype;
    if (config_dtype is! TensorDataType) {
      config_dtype = (config_dtype as Map<String, TensorDataType>)[fileName];
    }

    if (config_dtype && config_dtype != DataType.auto && DataType.values.contains(config_dtype)) {
      // Defined by the config, and is not "auto"
      dtype = config_dtype;
    } else {
      // Choose default dtype based on device, falling back to fp32
      dtype = DEFAULT_DEVICE_DTYPE_MAPPING[selectedDevice] ?? DataType.fp32;
    }

    // Choose default dtype based on device, falling back to fp32
    dtype = DEFAULT_DEVICE_DTYPE_MAPPING[selectedDevice] ?? DataType.fp32;
  }

  final DataType selectedDtype = dtype;

  if (!DEFAULT_DTYPE_SUFFIX_MAPPING.containsKey(selectedDtype)) {
    throw Exception('Invalid dtype: $selectedDtype. Should be one of: ${DataType.values.join(', ')}');
  } else if (selectedDtype == DataType.fp16 && selectedDevice == DeviceType.webgpu && !(await isWebGpuFp16Supported())) {
    throw Exception('The device ($selectedDevice) does not support fp16.');
  }

  // Only valid for models with a decoder
  final kv_cache_dtype_config = custom_config.kv_cache_dtype;
  final TensorDataType? kv_cache_dtype = kv_cache_dtype_config != null
      ? (kv_cache_dtype_config is TensorDataType
        ? kv_cache_dtype_config
        : kv_cache_dtype_config[selectedDtype] ?? TensorDataType.float32)
      : null;

  if (kv_cache_dtype != null && ![TensorDataType.float32, TensorDataType.float16].contains(kv_cache_dtype)) {
    throw Exception('Invalid kv_cache_dtype: $kv_cache_dtype. Should be one of: float32, float16');
  }

  final Map<String, dynamic> session_config = {
    'dtype': selectedDtype,
    'kv_cache_dtype': kv_cache_dtype,
    'device': selectedDevice,
  };

  // Construct the model file name
  final suffix = DEFAULT_DTYPE_SUFFIX_MAPPING[selectedDtype];
  final baseName = '$fileName$suffix.onnx';
  final modelFileName = '${options.subfolder}/$baseName';

  OrtSessionOptions session_options = options.session_options?.clone() ?? OrtSessionOptions();

  // Overwrite `executionProviders` if not specified
  session_options = session_options.cloneWith(
    providers: session_options.providers ?? executionProviders,
  );

  // Overwrite `freeDimensionOverrides` if specified in config and not set in session options
  final free_dimension_overrides = custom_config.free_dimension_overrides;
  if (free_dimension_overrides != null) {
    // TODO: flutter_onnxruntime doesn't have freeDimensionOverrides
    // session_options.freeDimensionOverrides ??= free_dimension_overrides;
  } else if (
    selectedDevice.value.startsWith('webnn')
      // TODO: flutter_onnxruntime doesn't have freeDimensionOverrides
      // && session_options.freeDimensionOverrides == null
  ) {
    // console.warn
    print(
      "WebNN does not currently support dynamic shapes and requires 'free_dimension_overrides' to be set in config.json, preferably as a field within config[\"transformers.js_config\"][\"device_config\"][\"$selectedDevice\"]. "
      "When 'free_dimension_overrides' is not set, you may experience significant performance degradation."
    );
  }

  final return_path = true; // Not on the web
  final bufferOrPathPromise = getModelFile(pretrained_model_name_or_path, modelFileName, true, options, return_path);

  // TODO: Handle onnx external data files and see if flutter_onnxruntime even supports this
  // Handle onnx external data files
  // ?? custom_config.use_external_data_format
  final dynamic use_external_data_format = options.use_external_data_format ?? custom_config.use_external_data_format;
  /** @type {Promise<string|{path: string, data: Uint8Array}>[]} */
  final List<Future<String>> externalDataPromises = [];
  if (use_external_data_format != null
      && (use_external_data_format is Map || use_external_data_format == true || use_external_data_format != 0)) {
    // bool or number
    dynamic external_data_format;
    if (use_external_data_format is Map) {
      if (use_external_data_format.containsKey(baseName)) {
        external_data_format = use_external_data_format[baseName];
      } else if (use_external_data_format.containsKey(fileName)) {
        external_data_format = use_external_data_format[fileName];
      } else {
        external_data_format = false;
      }
    } else {
      external_data_format = use_external_data_format;
    }

    // (false=0, true=1, number remains the same)
    final num_chunks = external_data_format is int
        ? external_data_format
        : (external_data_format ? 1 : 0);
    if (num_chunks > MAX_EXTERNAL_DATA_CHUNKS) {
      throw Exception('The number of external data chunks ($num_chunks) exceeds the maximum allowed value ($MAX_EXTERNAL_DATA_CHUNKS).');
    }

    for (int i=0; i < num_chunks; ++i) {
      final path = '${baseName}_data${i == 0 ? '' : '_$i'}';
      final fullPath = '${options.subfolder}/$path';
      externalDataPromises.add((() async {
        final data = await getModelFile(pretrained_model_name_or_path, fullPath, true, options, return_path);
        return path;
      })());
    }
  }
  // TODO: flutter_onnxruntime doesn't support externalData
  // else if (session_options.externalData != null) {
  //   externalDataPromises = session_options.externalData.map(async (ext) => {
  //     // if the external data is a string, fetch the file and replace the string with its content
  //     // @ts-expect-error TS2339
  //     if (typeof ext.data === "string") {
  //       // @ts-expect-error TS2339
  //       const ext_buffer = await getModelFile(pretrained_model_name_or_path, ext.data, true, options);
  //       // @ts-expect-error TS2698
  //       return { ...ext, data: ext_buffer };
  //     }
  //     return ext;
  //   });
  // }

  if (externalDataPromises.isNotEmpty) {
    final externalData = await Future.wait(externalDataPromises);
    // TODO: flutter_onnxruntime doesn't support externalData
    // if (!apis.IS_NODE_ENV) {
    //   session_options.externalData = externalData;
    // }
  }

  // This is ignored since web is not yet supported
  // if (selectedDevice == DeviceType.webgpu) {
  //   final shapes = getKeyValueShapes(options.config, prefix: 'present');
  //   if (shapes.keys.isNotEmpty && !isONNXProxy()) {
  //     // Only set preferredOutputLocation if shapes are present and we aren't proxying ONNX
  //     /** @type {Record<string, import('onnxruntime-common').Tensor.DataLocation>} */
  //     const preferredOutputLocation = {};
  //     for (const key in shapes) {
  //       preferredOutputLocation[key] = 'gpu-buffer';
  //     }
  //     session_options.preferredOutputLocation = preferredOutputLocation;
  //   }
  // }

  final buffer_or_path = await bufferOrPathPromise;

  return (buffer_or_path, session_options, session_config);
}

/// Helper function to create multiple InferenceSession objects.
///
/// @param {string} pretrained_model_name_or_path The path to the directory containing the model file.
/// @param {Record<string, string>} names The names of the model files to load.
/// @param {import('./utils/hub.js').PretrainedModelOptions} options Additional options for loading the model.
/// @returns {Promise<Record<string, any>>} A Promise that resolves to a dictionary of InferenceSession objects.
/// @private
Future<Map<String, OrtSession>> constructSessions(
  String pretrained_model_name_or_path,
  Map<String, String> names,
  PretrainedModelOptions options,
) async {
  return Map.fromEntries(await Future.wait(
    names.keys.map((name) async {
      final (buffer_or_path, session_options, session_config) = await getSession(
          pretrained_model_name_or_path, names[name]!, options);
      final session = await createInferenceSession(buffer_or_path, session_options, session_config);
      return MapEntry(name, session);
    }),
  ));
}

/// Helper function to load multiple optional configuration files
/// @param {string} pretrained_model_name_or_path The path to the directory containing the config file.
/// @param {Record<string, string>} names The names of the config files to load.
/// @param {import('./utils/hub.js').PretrainedModelOptions} options Additional options for loading the configs.
/// @returns {Promise<Record<string, any>>} A Promise that resolves to a dictionary of configuration objects.
/// @private
Future<Map<String, Map<String, dynamic>>> getOptionalConfigs(
    String pretrained_model_name_or_path,
    Map<String, String> names,
    PretrainedModelOptions options,
    ) async => Map.fromEntries(await Future.wait(
    names.keys.map((name) async {
      final config = await getModelJSON(pretrained_model_name_or_path, names[name]!, false, options);
      return MapEntry(name, config);
    })
));

/// Validate model inputs
/// @param {Object} session The InferenceSession object that will be run.
/// @param {Object} inputs The inputs to check.
/// @returns {Record<string, Tensor>} The checked inputs.
/// @throws {Error} If any inputs are missing.
/// @private
Future<Map<String, dynamic>> validateInputs(OrtSession session, Map<String, dynamic> inputs) async {
  /**
   * NOTE: Create either a shallow or deep copy based on `onnx.wasm.proxy`
   * @type {Record<string, Tensor>}
   */
  final Map<String, dynamic> checkedInputs = {};
  final missingInputs = [];
  for (final inputName in session.inputNames) {
    final tensor = inputs[inputName];
    // Rare case where one of the model's input names corresponds to a built-in
    // object name (e.g., toString), which would cause a simple (!tensor) check to fail,
    // because it's not undefined but a function.
    if (tensor is! Tensor) {
      missingInputs.add(inputName);
      continue;
    }
    // NOTE: When `env.wasm.proxy is true` the tensor is moved across the Worker
    // boundary, transferring ownership to the worker and invalidating the tensor.
    // So, in this case, we simply sacrifice a clone for it.
    checkedInputs[inputName] = isONNXProxy() ? await tensor.clone() : tensor;
  }
  if (missingInputs.isNotEmpty) {
    throw Exception('An error occurred during model execution: "Missing the following inputs: ${missingInputs.join(', ')}.');
  }

  final numInputsProvided = inputs.length;
  final numInputsNeeded = session.inputNames.length;
  if (numInputsProvided > numInputsNeeded) {
    // No missing inputs, but too many inputs were provided.
    // Warn the user and ignore the extra inputs.
    final ignored = inputs.keys.where((inputName) => !session.inputNames.contains(inputName));
    // console.warn
    print('WARNING: Too many inputs were provided ($numInputsProvided > $numInputsNeeded). The following inputs will be ignored: "${ignored.join(', ')}".');
  }

  return checkedInputs;
}

/// Executes an InferenceSession using the specified inputs.
/// NOTE: `inputs` must contain at least the input names of the model.
///  - If additional inputs are passed, they will be ignored.
///  - If inputs are missing, an error will be thrown.
///
/// @param {Object} session The InferenceSession object to run.
/// @param {Object} inputs An object that maps input names to input tensors.
/// @returns {Promise<Object>} A Promise that resolves to an object that maps output names to output tensors.
/// @private
Future<Map<String, Tensor<dynamic>>> sessionRun(OrtSession session, Map<String, dynamic> inputs) async {
  final checkedInputs = await validateInputs(session, inputs);
  try {
    // pass the original ort tensor
    final ortFeed = checkedInputs.map((k, v) => MapEntry(k, (v as Tensor).ort_tensor.ortValue));
    // run() => session.run(ortFeed);
    // final output = await ((apis.IS_BROWSER_ENV || apis.IS_WEBWORKER_ENV)
    //     ? (webInferenceChain = webInferenceChain.then(run))
    //     : run());
    final Map<String, OrtValue> output = await session.run(ortFeed);
    return replaceTensors(output);
  } catch (e) {
    // Error messages can be long (nested) and uninformative. For this reason,
    // we apply minor formatting to show the most important information
    final formatted = checkedInputs.map((k, tensor) {
      tensor = (tensor as Tensor);
      // Extract these properties from the underlying ORT tensor
      final unpacked = {
        'type': tensor.type,
        'dims': tensor.dims,
        'location': tensor.location,
      };
      if (unpacked['location'] != "gpu-buffer") {
        // Only return the data if it's not a GPU buffer
        unpacked['data'] = tensor.data;
      }
      return MapEntry(k, unpacked);
    });

    // This usually occurs when the inputs are of the wrong type.
    // console.error
    print('An error occurred during model execution: "$e".');
    // console.error
    print('Inputs given to model: $formatted');
    rethrow;
  }
}

/// Replaces ONNX Tensor objects with custom Tensor objects to support additional functions.
/// @param {Object} obj The object to replace tensor objects in.
/// @returns {Object} The object with tensor objects replaced by custom Tensor objects.
/// @private
Future<Map<String, Tensor>> replaceTensors(Map<String, OrtValue> obj) async {
  return Map.fromEntries(
      await Future.wait(
          obj.entries.map((e) async => MapEntry(e.key, await Tensor.createFromOrtValue(e.value)))
      )
  );
}

/// Converts an array or Tensor of integers to an int64 Tensor.
/// @param {any[]|Tensor} items The input integers to be converted.
/// @returns {Tensor} The int64 Tensor with the converted values.
/// @throws {Error} If the input array is empty or the input is a batched Tensor and not all sequences have the same length.
/// @private
Future<Tensor> toI64Tensor(dynamic items) async {
  if (items is Tensor) {
    return items;
  }

  // items is an array
  items = items as List;
  if (items.isEmpty) {
    throw Exception('items must be non-empty');
  }

  if (items[0] is List) {
    // batched
    if (items.any((x) => x.length != items[0].length)) {
      throw Exception("Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' and/or 'truncation=True' to have batched tensors with the same length.");
    }

    return await Tensor.create(
      TensorDataType.int64,
      items.expand((e) => e).toList(),
      [items.length, items[0].length],
    );
  } else {
    return await Tensor.create(
      TensorDataType.int64,
      items,
      [items.length, items[0].length],
    );
  }
}

/// Creates a boolean tensor with a single value.
/// @param {boolean} value The value of the tensor.
/// @returns {Tensor} The boolean tensor.
/// @private
Future<Tensor<bool>> boolTensor(bool value) async {
  return await Tensor.create<bool>(TensorDataType.bool, [value], [1]);
}

abstract class AbstractPreTrainedModel {
  /// The inference sessions for the model.
  Map<String, OrtSession> get sessions;

  /// PretrainedConfig config The model configuration.
  PretrainedConfig? get config;

  /// Adds past key values to the decoder feeds object. If pastKeyValues is null, creates new tensors for past key values.
  ///
  /// @param {Object} decoderFeeds The decoder feeds object to add past key values to.
  /// @param {Object} pastKeyValues An object containing past key values.
  Future<void> addPastKeyValues(Map<String, dynamic> decoderFeeds, Map<String, dynamic>? pastKeyValues);

  Future<(Tensor, Tensor)> get_rope_index(
      Tensor input_ids,
      Tensor image_grid_thw,
      Tensor video_grid_thw,
      Tensor attention_mask,
      ) async {
    throw UnimplementedError('get_rope_index');
  }

  /// TODO: Figure out the method definition
  _merge_input_ids_with_audio_features(Map<String, dynamic> params) {
    throw UnimplementedError('_merge_input_ids_with_audio_features should be implemented in subclass.');
  }

  Future<({Tensor inputs_embeds, dynamic attention_mask})> _merge_input_ids_with_image_features(Map<String, dynamic> kwargs) async {
    throw UnimplementedError('_merge_input_ids_with_image_features should be implemented in subclass.');
  }

  encode(Map<String, dynamic> params) async {
    throw UnimplementedError('encode should be implemented in subclass.');
  }

  decode(Map<String, dynamic> params) async {
    throw UnimplementedError('encode should be implemented in subclass.');
  }
}

mixin PreTrainedModelMixin on AbstractPreTrainedModel {
  /// Perform forward pass on the seq2seq model (both encoder and decoder).
  /// @param {Object} self The seq2seq model object.
  /// @param {Object} model_inputs The input object for the model containing encoder and decoder inputs.
  /// @returns {Promise<Seq2SeqLMOutput>} Promise that resolves with the output of the seq2seq model.
  /// @private
  Future<Map<String, Tensor>> seq2seqForward(Map<String, dynamic> model_inputs, [bool _ = false]) async {
    model_inputs = { ...model_inputs };
    var encoder_outputs = model_inputs.remove('encoder_outputs'),
        input_ids = model_inputs.remove('input_ids'),
        decoder_input_ids = model_inputs.remove('decoder_input_ids');
    final other_decoder_inputs = model_inputs;
    // Encode if needed
    if (encoder_outputs == null) {
      final encoder_inputs = pick(model_inputs, sessions['model']!.inputNames);
      // Encoder outputs are not given, so we must compute them.
      encoder_outputs = (await encoderForward(encoder_inputs))['last_hidden_state'];
    }

    other_decoder_inputs['input_ids'] = decoder_input_ids;
    other_decoder_inputs['encoder_hidden_states'] = encoder_outputs;

    if (sessions['decoder_model_merged']!.inputNames.contains('encoder_attention_mask')) {
      other_decoder_inputs['encoder_attention_mask'] = model_inputs['attention_mask'];
    }

    final decoderResults = await decoderForward(other_decoder_inputs, true);

    return decoderResults;
  }

  /// Forward pass of an encoder model.
  /// @param {Object} self The encoder model.
  /// @param {Object} model_inputs The input data to be used for the forward pass.
  /// @returns {Promise<Object>} The model's outputs.
  /// @private
  Future<Map<String, Tensor>> encoderForward(Map<String, dynamic> model_inputs, [bool _ = false]) async {
    final session = sessions['model']!;
    final encoderFeeds = pick(model_inputs, session.inputNames);

    if (session.inputNames.contains('inputs_embeds') && encoderFeeds['inputs_embeds'] == null) {
      if (model_inputs['input_ids'] == null) {
        throw Exception('Both `input_ids` and `inputs_embeds` are missing in the model inputs.');
      }
      encoderFeeds['inputs_embeds'] = await encode_text({ 'input_ids': model_inputs['input_ids'] });
    }
    if (session.inputNames.contains('token_type_ids') && encoderFeeds['token_type_ids'] == null) {
      if (encoderFeeds['input_ids'] == null) {
        throw Exception('Both `input_ids` and `token_type_ids` are missing in the model inputs.');
      }
      // Assign default `token_type_ids` (all zeroes) to the `encoderFeeds` if the model expects it,
      // but they weren't created by the tokenizer.
      encoderFeeds['token_type_ids'] = await zeros_like(encoderFeeds['input_ids']);
    }
    if (session.inputNames.contains('pixel_mask') && encoderFeeds['pixel_mask'] == null) {
      if (encoderFeeds['pixel_values'] == null) {
        throw Exception('Both `pixel_values` and `pixel_mask` are missing in the model inputs.');
      }
      // Assign default `pixel_mask` (all ones) to the `encoderFeeds` if the model expects it,
      // but they weren't created by the processor.
      final dims = (encoderFeeds['pixel_values'] as Tensor).dims;
      encoderFeeds['pixel_mask'] = await ones([dims[0], dims[2], dims[3]]);
    }

    return await sessionRun(session, encoderFeeds);
  }

  Future<Map<String, Tensor>> autoEncoderForward(Map<String, dynamic> model_inputs, [bool _ = false]) async {
    final encoded = await encode(model_inputs);
    final decoded = await decode(encoded);
    return decoded;
  }

  /// Forward pass of a decoder model.
  /// @param {Object} self The decoder model.
  /// @param {Object} model_inputs The input data to be used for the forward pass.
  /// @returns {Promise<Object>} The logits and past key values.
  /// @private
  Future<Map<String, Tensor>> decoderForward(Map<String, dynamic> model_inputs, [bool is_encoder_decoder = false]) async {
    final session = sessions[is_encoder_decoder ? 'decoder_model_merged' : 'model']!;

    model_inputs = { ...model_inputs };
    final past_key_values = model_inputs.remove('past_key_values');
    final Map<String, dynamic> new_model_inputs = model_inputs;

    if (session.inputNames.contains('use_cache_branch')) {
      // TODO: Might need to better handle the bool value js: !!past_key_values
      new_model_inputs['use_cache_branch'] = await boolTensor(past_key_values != null || past_key_values == true);
    }
    if (session.inputNames.contains('position_ids')
        && new_model_inputs['attention_mask'] != null && new_model_inputs['position_ids'] == null) {
      // NOTE: Handle a special case for paligemma/gemma3 models, where positions are 1-indexed
      final start_index = ['paligemma', 'gemma3_text', 'gemma3'].contains(config?.model_type) ? 1 : 0;
      new_model_inputs['position_ids'] = await createPositionIds(new_model_inputs, past_key_values, start_index);
    }

    // Unpack the `past_key_values` object into model inputs
    await addPastKeyValues(new_model_inputs, past_key_values);

    // Select only the inputs that are needed for the current session
    final fixed = pick(new_model_inputs, session.inputNames);
    return await sessionRun(session, fixed);
  }

  /// Abstract forward pass function for image-text-to-text or audio-text-to-text models.
  /// @param {Object} self The model object.
  /// @param {Object} params Additional parameters.
  /// @param {Function} [params.encode_function] The function to encode the modality values.
  /// @param {Function} [params.merge_function] The function to merge the modality features with the input embeddings.
  /// @param {string} [params.modality_input_name] The modality input name.
  /// @param {string} [params.modality_output_name] The modality output name.
  /// @param {Tensor} [params.input_ids=null]
  /// @param {Tensor} [params.attention_mask=null]
  /// @param {Tensor} [params.position_ids=null]
  /// @param {Tensor} [params.inputs_embeds=null]
  /// @param {Tensor} [params.past_key_values=null]
  /// @param {Object} [params.generation_config=null]
  /// @param {Object} [params.logits_processor=null]
  /// @returns {Promise<Tensor>} The model's output tensor
  /// @private
  Future<Map<String, Tensor<dynamic>>> genericTextToTextForward(Map<String, dynamic> params) async {
    params = { ...params };

    var
      // Generic parameters:
      encode_function = params.remove('encode_function'),
      merge_function = params.remove('merge_function'),
      modality_input_name = params.remove('modality_input_name'),
      modality_output_name = params.remove('modality_output_name'),

      // Produced by the tokenizer/processor:
      input_ids = params.remove('input_ids'),
      attention_mask = params.remove('attention_mask'),

      // Used during generation:
      position_ids = params.remove('position_ids'),
      inputs_embeds = params.remove('inputs_embeds'),
      past_key_values = params.remove('past_key_values'),

      // Generic generation parameters
      generation_config = params.remove('generation_config'),
      logits_processor = params.remove('logits_processor'),

      // Additional parameters
      kwargs = params;

    final modality_values = kwargs[modality_input_name];
    if (inputs_embeds == null) {
      // 1. Extract the text embeddings.
      inputs_embeds = await encode_text({ 'input_ids': input_ids, ...kwargs });

      // 2. Possibly, merge text and modality values
      if (modality_values != null && input_ids.dims[1] != 1) {
        final modality_features = await encode_function(<String, dynamic>{
          // Pass the modality values under its expected key.
          // The caller knows whether this is audio or image.
          modality_input_name: modality_values,
          ...kwargs
        });

        final merged = await merge_function(<String, dynamic>{
          modality_output_name: modality_features,
          'inputs_embeds': inputs_embeds,
          'input_ids': input_ids,
          'attention_mask': attention_mask,
        });
        inputs_embeds = merged.inputs_embeds;
        attention_mask = merged.attention_mask;
      } else if (past_key_values != null && modality_values != null && input_ids.dims[1] == 1) {
        // This branch handles the cache case.
        final target_length = input_ids.dims[1]; // always 1
        final past_key_values_dim = ((past_key_values as Map).values.first as Tensor).dims;
        final past_length = past_key_values_dim[past_key_values_dim.length - 2];

        attention_mask = await cat([
          await ones([input_ids.dims[0], past_length]),
          await (attention_mask as Tensor).slice([null, [attention_mask.dims[1] - target_length, attention_mask.dims[1]]]),
        ], 1);
      }
    }

    if (position_ids == null) {
      if (config?.model_type == 'qwen2_vl') {
        // Special case for qwen2_vl models
        final image_grid_thw = kwargs['image_grid_thw'],
            video_grid_thw = kwargs['video_grid_thw'];
        (position_ids, _) = await get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask);
      }
    }

    // 3. Call the decoder forward using the updated inputs.
    final outputs = await decoderForward({
      'inputs_embeds': inputs_embeds,
      'past_key_values': past_key_values,
      'attention_mask': attention_mask,
      'position_ids': position_ids,
      'generation_config': generation_config,
      'logits_processor': logits_processor,
    }, true);
    return outputs;
  }

  /// Forward pass of an audio-text-to-text model.
  /// @param {Object} self The audio-text-to-text model.
  /// @param {Object} params The inputs for the audio-text-to-text forward pass.
  /// @returns {Promise<Tensor>} The model's output tensor.
  /// @private
  Future<Map<String, Tensor<dynamic>>> audioTextToTextForward(Map<String, dynamic> params, [bool _ = false]) async {
    return await genericTextToTextForward({
      ...params,
      'modality_input_name': 'audio_values',
      'modality_output_name': 'audio_features',
      'encode_function': encode_audio,
      'merge_function': _merge_input_ids_with_audio_features,
    });
  }

  /// Forward pass of an image-text-to-text model.
  /// @param {Object} self The image-text-to-text model.
  /// @param {Object} params The inputs for the image-text-to-text forward pass.
  /// @returns {Promise<Tensor>} The model's output tensor.
  /// @private
  Future<Map<String, Tensor<dynamic>>> imageTextToTextForward(Map<String, dynamic> params, [bool _ = false]) async {
    return await genericTextToTextForward({
      ...params,
      'modality_input_name': 'pixel_values',
      'modality_output_name': 'image_features',
      'encode_function': encode_image,
      'merge_function': _merge_input_ids_with_image_features,
    });
  }

  Future<Map<String, dynamic>> decoder_prepare_inputs_for_generation(
    List<dynamic> input_ids,
    Map<String, dynamic> model_inputs,
    generation_config,
  ) async {
    if (model_inputs['past_key_values'] != null) {
      final past_key_value_dims = List<Tensor>.from(model_inputs['past_key_values'].values).first.dims;
      final past_length = past_key_value_dims[past_key_value_dims.length - 2];
      final Tensor input_ids = model_inputs['input_ids'];
      final Tensor? attention_mask = model_inputs['attention_mask'];

      // Keep only the unprocessed tokens:
      // 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
      // some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
      // input)
      if (attention_mask != null && attention_mask.dims[1] > input_ids.dims[1]) {
        // NOTE: not needed since we only pass the generated tokens to the next forward pass
        // const offset = -(attention_mask.dims[1] - past_length);
        // model_inputs.input_ids = input_ids.slice(null, [offset, null]);
      }
      // 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens.
      // We can discard input_ids based on the past_length.
      else if (past_length < input_ids.dims[1]) {
        // NOTE: Required for phi models.
        // See https://github.com/huggingface/transformers/issues/30809#issuecomment-2111918479 for more information.
        model_inputs['input_ids'] = await input_ids.slice([null, [past_length, null]]);
      }
      // 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
      else {
        if (
        // NOTE: Only used by VLMs (!= so that null matches undefined)
        config?.image_token_index != null &&
            // Equivalent to `self.config.image_token_index in input_ids` (== so that int matches bigint)
            input_ids.data.any((x) => x == config?.image_token_index)
        ) {
          // TODO: Support multiple image tokens
          final int? num_image_tokens = config?.num_image_tokens;
          if (num_image_tokens == null) {
            throw Exception('`num_image_tokens` is missing in the model configuration.');
          }

          final num_new_tokens = input_ids.dims[1] - (past_length - num_image_tokens);
          model_inputs['input_ids'] = await input_ids.slice([null, [-num_new_tokens, null]]);

          // TODO: The attention mask should be formed from the attention mask passed in model_inputs
          model_inputs['attention_mask'] = await ones([1, past_length + num_new_tokens]);
        }
      }
    }

    return model_inputs;
  }

  Future<Map<String, dynamic>> encoder_decoder_prepare_inputs_for_generation(
    List<List<int>> input_ids,
    Map<String, dynamic> model_inputs,
    generation_config,
  ) async {
    if (model_inputs['past_key_values'] != null) {
      input_ids = input_ids.map((x) => [x.last]).toList();
    }

    return {
      ...model_inputs,
      'decoder_input_ids': await toI64Tensor(input_ids),
    };
  }

  Future<Map<String, dynamic>> multimodal_text_to_text_prepare_inputs_for_generation(
      List<List<int>> input_ids,
      Map<String, dynamic> model_inputs,
      generation_config,
      ) async {
    if (config?.is_encoder_decoder != true) {
      return await encoder_decoder_prepare_inputs_for_generation(input_ids, model_inputs, generation_config);
    } else {
      return await decoder_prepare_inputs_for_generation(input_ids, model_inputs, generation_config);
    }
  }

  Future<Map<String, dynamic>> multimodality_prepare_inputs_for_generation(
    List<dynamic> input_ids,
    Map<String, dynamic> model_inputs,
    generation_config,
  ) async {
    final has_past_key_values = model_inputs['past_key_values'] != null;

    if (generation_config.guidance_scale != null && generation_config.guidance_scale > 1) {
      if (has_past_key_values) {
        model_inputs['input_ids'] = await cat([
          model_inputs['input_ids'] as Tensor,
          model_inputs['input_ids'] as Tensor,
        ], 0);
        // NOTE: attention_mask handled in generation
      } else {
        model_inputs['input_ids'] = await cat([
          model_inputs['input_ids'] as Tensor,
          await full_like(model_inputs['input_ids'], generation_config['pad_token_id']),
        ], 0);
        model_inputs['attention_mask'] = await cat([
          model_inputs['attention_mask'] as Tensor,
          await full_like(model_inputs['attention_mask'], 0),
        ], 0);
      }
    }

    if (has_past_key_values || model_inputs['pixel_values'] == null) {
      model_inputs['pixel_values'] = await full([0, 0, 3, 384, 384], 1.0);
    }

    if (has_past_key_values) {
      const int num_img_tokens = 0;
      const int num_text_tokens = 1;
      const int has_image = num_img_tokens > 0 ? 1 : 0;

      const batch_size = 1;
      model_inputs['images_seq_mask'] = await Tensor.create(
        TensorDataType.bool,
        [
          ...List<bool>.filled(num_text_tokens, false),
          ...List<bool>.filled(num_img_tokens, true),
        ],
        [batch_size, num_img_tokens + num_text_tokens],
      );
      model_inputs['images_emb_mask'] = await Tensor.create(
        TensorDataType.bool,
        List<bool>.filled(num_img_tokens, has_image == 1),
        [batch_size, 1, num_img_tokens],
      );
    }
    return model_inputs;
  }

  Future<Tensor> encode_image(Map<String, dynamic> params) async {
    // image_inputs === { pixel_values }
    final features = (await sessionRun(sessions['vision_encoder']!, { 'pixel_values': params['pixel_values'] }))['image_features']!;
    if (config?.num_image_tokens == null) {
      // console.warn
      print(
          'The number of image tokens was not set in the model configuration. '
              'Setting it to the number of features detected by the vision encoder (${features.dims[1]}).'
      );
      config?.num_image_tokens = features.dims[1];
    }
    return features;
  }

  Future<Tensor> encode_text(Map<String, dynamic> params) async {
    // text_inputs === { input_ids, attention_mask }
    return (await sessionRun(sessions['embed_tokens']!, { 'input_ids': params['input_ids'] }))['inputs_embeds']!;
  }

  Future<Tensor> encode_audio(Map<String, dynamic> params) async {
    // audio_inputs === { audio_values }
    return (await sessionRun(sessions['audio_encoder']!, { 'audio_values': params['audio_values'] }))['audio_features']!;
  }
}

Future<({Tensor inputs_embeds, dynamic attention_mask})> default_merge_input_ids_with_features(Map<String, dynamic> kwargs) async {
  final modality_token_id = kwargs['modality_token_id'];
  final Tensor inputs_embeds = kwargs['inputs_embeds'];
  final Tensor modality_features = kwargs['modality_features'];
  final Tensor input_ids = kwargs['input_ids'];
  final attention_mask = kwargs['attention_mask'];

  final token_positions = input_ids.tolist().map((ids) {
    List<int> acc = [];
    for (int idx = 0; idx < ids.length; idx++) {
      final x = ids[idx];

      if (x == modality_token_id) acc.add(idx);
    }
    return acc;
  }).toList();
  final n_tokens = token_positions.fold(0, (acc, x) => acc + x.length);
  final n_features = modality_features.dims[0];
  if (n_tokens != n_features) {
    throw Exception('Number of tokens and features do not match: tokens: $n_tokens, features $n_features');
  }

  // Equivalent to performing a masked_scatter
  // TODO: This NEEDS to be improved it is REALLY BAD
  int img = 0;
  for (int i = 0; i < token_positions.length; ++i) {
    final tokens = token_positions[i];
    final embeds = await inputs_embeds[i];
    for (int j = 0; j < tokens.length; ++j) {
      await (await embeds[tokens[j]]).updateData(((await modality_features[img++]).data));
    }
  }
  return (inputs_embeds: inputs_embeds, attention_mask: attention_mask);
}

Future<({Tensor inputs_embeds, dynamic attention_mask})> default_merge_input_ids_with_image_features(Map<String, dynamic> kwargs) async {
  return await default_merge_input_ids_with_features({
    'modality_token_id': kwargs['image_token_id'],
    'inputs_embeds': kwargs['inputs_embeds'],
    'modality_features': kwargs['image_features'],
    'input_ids': kwargs['input_ids'],
    'attention_mask': kwargs['attention_mask'],
  });
}

/// Helper function to perform the following:
/// ```python
/// x = attention_mask.long().cumsum(-1) - 1
/// x.masked_fill_(attention_mask == 0, 1)
/// ```
/// @param {Tensor} attention_mask
/// @returns {{data: BigInt64Array, dims: number[]}}
(List<int>, List<int>) cumsum_masked_fill(Tensor attention_mask, [int start_index = 0]) {
  final [int bz, int seq_len] = attention_mask.dims;
  final List<int> attn_mask_data = List<int>.from(attention_mask.data);

  final List<int> data = List<int>.filled(attn_mask_data.length, 0);
  for (int i=0; i < bz; ++i) {
    final start = i * seq_len;
    int sum = start_index;
    for (int j=0; j < seq_len; ++j) {
      final index = start + j;
      if (attn_mask_data[index] == 0) {
        data[index] = 1;
      } else { // === 1n
        data[index] = sum;
        sum += attn_mask_data[index];
      }
    }
  }
  return (data, attention_mask.dims);
}

/// If the model supports providing position_ids, we create position_ids on the fly for batch generation,
/// by computing the cumulative sum of the attention mask along the sequence length dimension.
///
/// Equivalent to:
/// ```python
/// position_ids = attention_mask.long().cumsum(-1) - 1
/// position_ids.masked_fill_(attention_mask == 0, 1)
/// if past_key_values:
///     position_ids = position_ids[:, -input_ids.shape[1] :]
/// ```
Future<Tensor<int>> createPositionIds(Map<String, dynamic> model_inputs, [past_key_values = null, int start_index = 0]) async {
  final input_ids = model_inputs['input_ids'],
      inputs_embeds = model_inputs['inputs_embeds'],
      attention_mask = model_inputs['attention_mask'];

  final (data, dims) = cumsum_masked_fill(attention_mask, start_index);
  Tensor<int> position_ids = await Tensor.create(TensorDataType.int64, data, dims);
  if (past_key_values != null) {
    final offset = -(input_ids ?? inputs_embeds).dims[1];
    position_ids = await position_ids.slice([null, [offset, null]]);
  }
  return position_ids;
}

//////////////////////////////////////////////////

//////////////////////////////////////////////////
/// Dart does not support extending static methods from a class. Therefore, this
/// is handled with the help of some wrapper classes and runtime checks. This
/// class is used as a way to access a particular classes methods, properties,
/// etc. as needed for the [PreTrainedModel] class.
class PreTrainedModelReflection {
  final Type type;

  final PreTrainedModel Function(
    PretrainedConfig? config,
    Map<String, OrtSession> sessions,
    Map<String, Map<String, dynamic>> configs,
  ) constructor;

  PreTrainedModelReflection({
    required this.type,
    required this.constructor,
  });
}

/// A base class for pre-trained models that provides the model configuration and an ONNX session.
class PreTrainedModel extends AbstractPreTrainedModel with PreTrainedModelMixin {
  static final PreTrainedModelReflection _reflection = PreTrainedModelReflection(
    type: PreTrainedModel,
    constructor: constructor,
  );

  final PreTrainedModelReflection reflection;

  final String main_input_name = 'input_ids';

  final List<String> forward_params = ['input_ids', 'attention_mask'];

  /// PretrainedConfig config The model configuration.
  @override
  final PretrainedConfig? config;

  /// The inference sessions for the model.
  @override
  final Map<String, OrtSession> sessions;

  /// Additional configuration files (e.g., generation_config.json).
  final Map<String, Map<String, dynamic>> configs;

  bool can_generate = false;

  late Future<Map<String, Tensor>> Function(
    Map<String, dynamic> model_inputs,
    [bool is_encoder_decoder]
  ) _forward;

  /// If [_prepare_inputs_embeds] is implemented by the subclass then this must
  /// be overridden and set to true. Since, unlike JavaScript and Python, the
  /// method must be implemented.
  final bool _is_prepare_inputs_embeds_supported = false;

  Future<Map<String, dynamic>> Function(
    List<List<int>> input_ids,
    Map<String, dynamic> model_inputs,
    dynamic generation_config,
  )? _prepare_inputs_for_generation;

  late TransformersJSConfig custom_config;

  static PreTrainedModel constructor(
    PretrainedConfig? config,
    Map<String, OrtSession> sessions,
    Map<String, Map<String, dynamic>> configs,
  ) => PreTrainedModel(_reflection, config, sessions, configs);

  /// Creates a new instance of the `PreTrainedModel` class.
  /// @param {import('./configs.js').PretrainedConfig} config The model configuration.
  /// @param {Record<string, any>} sessions The inference sessions for the model.
  /// @param {Record<string, Object>} configs Additional configuration files (e.g., generation_config.json).
  PreTrainedModel(this.reflection, this.config, this.sessions, this.configs) {
    final modelName = MODEL_CLASS_TO_NAME_MAPPING[reflection.type];
    final modelType = MODEL_TYPE_MAPPING[modelName];

    switch (modelType) {
      case MODEL_TYPES.DecoderOnly:
        can_generate = true;
        _forward = decoderForward;
        _prepare_inputs_for_generation = decoder_prepare_inputs_for_generation;
        break;
      case MODEL_TYPES.Seq2Seq:
      case MODEL_TYPES.Vision2Seq:
      case MODEL_TYPES.Musicgen:
        can_generate = true;

        _forward = seq2seqForward;
        _prepare_inputs_for_generation = encoder_decoder_prepare_inputs_for_generation;
        break;

      case MODEL_TYPES.EncoderDecoder:
        _forward = seq2seqForward;
        break;
      case MODEL_TYPES.ImageTextToText:
        can_generate = true;
        _forward = imageTextToTextForward;
        _prepare_inputs_for_generation = multimodal_text_to_text_prepare_inputs_for_generation;
        break;
      case MODEL_TYPES.AudioTextToText:
        can_generate = true;
        _forward = audioTextToTextForward;
        _prepare_inputs_for_generation = multimodal_text_to_text_prepare_inputs_for_generation;
        break;
      case MODEL_TYPES.Phi3V:
        can_generate = true;
        _prepare_inputs_for_generation = multimodal_text_to_text_prepare_inputs_for_generation;
        break;
      case MODEL_TYPES.MultiModality:
        can_generate = true;
        _prepare_inputs_for_generation = multimodality_prepare_inputs_for_generation;
        break;
      case MODEL_TYPES.AutoEncoder:
        _forward = autoEncoderForward;
        break;
      default:
        // should be MODEL_TYPES.EncoderOnly
        _forward = encoderForward;
        break;
    }

    if (can_generate) {
      forward_params.add('past_key_values');
    }

    custom_config = TransformersJSConfig.fromJson(config?['transformers.js_config'] ?? {});
  }

  /// Disposes of all the ONNX sessions that were created during inference.
  /// @returns {Promise<unknown[]>} An array of promises, one for each ONNX session that is being disposed.
  /// @todo Use https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/FinalizationRegistry
  Future<List<void>> dispose() async {
    return await Future.wait(sessions.values.map((session) => session.dispose()));
  }

  /// Helper method to setup [from_pretrained] with the correct [reflection] inserted.
  static Future<PreTrainedModel> Function(
    String pretrained_model_name_or_path,
    PretrainedModelOptions? options,
  ) setup_from_pretrained(PreTrainedModelReflection reflection) => (
      String pretrained_model_name_or_path,
      PretrainedModelOptions? options,
    ) => PreTrainedModel.from_pretrained(
      reflection,
      pretrained_model_name_or_path,
      options,
    );

  /// Get the default [from_pretrained] with [PreTrainedModel]'s own reflection
  /// inserted.
  static Future<PreTrainedModel> Function(
    String pretrained_model_name_or_path,
    PretrainedModelOptions? options,
  ) get default_from_pretrained => setup_from_pretrained(_reflection);

  /// Instantiate one of the model classes of the library from a pretrained model.
  ///
  /// The model class to instantiate is selected based on the `model_type` property of the config object
  /// (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible)
  ///
  /// @param {string} pretrained_model_name_or_path The name or path of the pretrained model. Can be either:
  /// - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  ///   Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
  ///   user or organization name, like `dbmdz/bert-base-german-cased`.
  /// - A path to a *directory* containing model weights, e.g., `./my_model_directory/`.
  /// @param {import('./utils/hub.js').PretrainedModelOptions} options Additional options for loading the model.
  ///
  /// @returns {Promise<PreTrainedModel>} A new instance of the `PreTrainedModel` class.
  static Future<PreTrainedModel> from_pretrained(
    PreTrainedModelReflection reflection,
    String pretrained_model_name_or_path,
    PretrainedModelOptions? options,
  ) async {
    options ??= PretrainedModelOptions();

    final modelName = MODEL_CLASS_TO_NAME_MAPPING[reflection.type];
    final modelType = MODEL_TYPE_MAPPING[modelName];

    final config = await AutoConfig.from_pretrained(pretrained_model_name_or_path, options);
    options.config = config;

    List<dynamic> info;
    if (modelType == MODEL_TYPES.DecoderOnly) {
      info = await Future.wait([
        constructSessions(pretrained_model_name_or_path, {
          'model': options.model_file_name ?? 'model',
        }, options),
        getOptionalConfigs(pretrained_model_name_or_path, {
          'generation_config': 'generation_config.json',
        }, options),
      ]);
    } else if (modelType == MODEL_TYPES.Seq2Seq || modelType == MODEL_TYPES.Vision2Seq) {
      info = await Future.wait([
        constructSessions(pretrained_model_name_or_path, {
          'model': 'encoder_model',
          'decoder_model_merged': 'decoder_model_merged',
        }, options),
        getOptionalConfigs(pretrained_model_name_or_path, {
          'generation_config': 'generation_config.json',
        }, options),
      ]);
    } else if (modelType == MODEL_TYPES.MaskGeneration) {
      info = await Future.wait([
        constructSessions(pretrained_model_name_or_path, {
          'model': 'vision_encoder',
          'prompt_encoder_mask_decoder': 'prompt_encoder_mask_decoder',
        }, options),
      ]);
    } else if (modelType == MODEL_TYPES.EncoderDecoder) {
      info = await Future.wait([
        constructSessions(pretrained_model_name_or_path, {
          'model': 'encoder_model',
          'decoder_model_merged': 'decoder_model_merged',
        }, options),
      ]);
    } else if (modelType == MODEL_TYPES.ImageTextToText) {
      const sessions = {
        'embed_tokens': 'embed_tokens',
        'vision_encoder': 'vision_encoder',
        'decoder_model_merged': 'decoder_model_merged',
      };
      if (config.is_encoder_decoder) {
        sessions['model'] = 'encoder_model';
      }
      info = await Future.wait([
        constructSessions(pretrained_model_name_or_path, sessions, options),
        getOptionalConfigs(pretrained_model_name_or_path, {
          'generation_config': 'generation_config.json',
        }, options),
      ]);
    } else if (modelType == MODEL_TYPES.AudioTextToText) {
      const sessions = {
        'embed_tokens': 'embed_tokens',
        'audio_encoder': 'audio_encoder',
        'decoder_model_merged': 'decoder_model_merged',
      };
      info = await Future.wait([
        constructSessions(pretrained_model_name_or_path, sessions, options),
        getOptionalConfigs(pretrained_model_name_or_path, {
          'generation_config': 'generation_config.json',
        }, options),
      ]);
    } else if (modelType == MODEL_TYPES.Musicgen) {
      info = await Future.wait([
        constructSessions(pretrained_model_name_or_path, {
          'model': 'text_encoder',
          'decoder_model_merged': 'decoder_model_merged',
          'encodec_decode': 'encodec_decode',
        }, options),
        getOptionalConfigs(pretrained_model_name_or_path, {
          'generation_config': 'generation_config.json',
        }, options),
      ]);
    } else if (modelType == MODEL_TYPES.MultiModality) {
      info = await Future.wait([
        constructSessions(pretrained_model_name_or_path, {
          'prepare_inputs_embeds': 'prepare_inputs_embeds',
          'model': 'language_model',
          'lm_head': 'lm_head',
          'gen_head': 'gen_head',
          'gen_img_embeds': 'gen_img_embeds',
          'image_decode': 'image_decode',
        }, options),
        getOptionalConfigs(pretrained_model_name_or_path, {
          'generation_config': 'generation_config.json',
        }, options),
      ]);
    } else if (modelType == MODEL_TYPES.Phi3V) {
      info = await Future.wait([
        constructSessions(pretrained_model_name_or_path, {
          'prepare_inputs_embeds': 'prepare_inputs_embeds',
          'model': 'model',
          'vision_encoder': 'vision_encoder',
        }, options),
        getOptionalConfigs(pretrained_model_name_or_path, {
          'generation_config': 'generation_config.json',
        }, options),
      ]);
    } else if (modelType == MODEL_TYPES.AutoEncoder) {
      info = await Future.wait([
        constructSessions(pretrained_model_name_or_path, {
          'encoder_model': 'encoder_model',
          'decoder_model': 'decoder_model',
        }, options),
      ]);
    } else { // should be MODEL_TYPES.EncoderOnly
      if (modelType != MODEL_TYPES.EncoderOnly) {
        final type = modelName ?? config.model_type;
        if (type != 'custom') {
          // console.warn
          print("Model type for '$type' not found, assuming encoder-only architecture. Please report this at $GITHUB_ISSUE_URL.");
        }
      }
      info = await Future.wait([
        constructSessions(pretrained_model_name_or_path, {
          'model': options.model_file_name ?? 'model',
        }, options),
      ]);
    }

    final Map<String, OrtSession> sessions = info.first;
    final Map<String, Map<String, dynamic>> configs = info.length == 1 ? {} : info.last;

    return reflection.constructor(config, sessions, configs);
  }

  /// Runs the model with the provided inputs
  /// @param {Object} model_inputs Object containing input tensors
  /// @returns {Promise<Object>} Object containing output tensors
  Future<Map<String, Tensor>> call(Map<String, dynamic> model_inputs) async => await forward(model_inputs);

  /// Forward method for a pretrained model. If not overridden by a subclass, the correct forward method
  /// will be chosen based on the model type.
  /// @param {Object} model_inputs The input data to the model in the format specified in the ONNX model.
  /// @returns {Promise<Object>} The output data from the model in the format specified in the ONNX model.
  /// @throws {Error} This method must be implemented in subclasses.
  Future<Map<String, Tensor>> forward(Map<String, dynamic> model_inputs) async => await _forward(model_inputs);

  /// Get the model's generation config, if it exists.
  /// @returns {GenerationConfig|null} The model's generation config if it exists, otherwise `null`.
  GenerationConfig? get generation_config {
    final generation_config = configs['generation_config'];
    return generation_config == null ? null : GenerationConfig.fromJson(generation_config);
  }

  /// This function returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`]
  /// instances used for multinomial sampling.
  /// @param {GenerationConfig} generation_config The generation config.
  /// @returns {LogitsProcessorList} generation_config
  LogitsProcessorList _get_logits_warper(GenerationConfig generation_config) {
    // instantiate warpers list
    final warpers = LogitsProcessorList();

    if (generation_config.temperature != null && generation_config.temperature != 1.0) {
      warpers.add(TemperatureLogitsWarper(generation_config.temperature));
    }
    if (generation_config.top_k != null && generation_config.top_k != 0) {
      // TODO: add min_tokens_to_keep
      warpers.add(TopKLogitsWarper(generation_config.top_k));
    }
    if (generation_config.top_p != null && generation_config.top_p < 1.0) {
      // TODO: add min_tokens_to_keep
      warpers.add(TopPLogitsWarper(generation_config.top_p));
    }

    return warpers;
  }

  /// @param {GenerationConfig} generation_config
  /// @param {number} input_ids_seq_length The starting sequence length for the input ids.
  /// @returns {LogitsProcessorList}
  /// @private
  LogitsProcessorList _get_logits_processor(
      GenerationConfig generation_config,
      int input_ids_seq_length,
      // encoder_input_ids, TODO
      // prefix_allowed_tokens_fn, TODO
      dynamic logits_processor,
      ) {
    final processors = LogitsProcessorList();

    // Below was commented out in transformers.js:
    // if (generation_config.diversity_penalty !== null && generation_config.diversity_penalty > 0.0) {
    //     processors.push(new HammingDiversityLogitsProcessor(
    //         generation_config.diversity_penalty,
    //         generation_config.num_beams,
    //         generation_config.num_beam_groups
    //     ));
    // }

    // Below was commented out in transformers.js:
    // if (generation_config.encoder_repetition_penalty !== null && generation_config.encoder_repetition_penalty !== 1.0) {
    //     processors.push(new EncoderRepetitionPenaltyLogitsProcessor(
    //         generation_config.encoder_repetition_penalty,
    //         encoder_input_ids
    //     ));
    // }

    if (generation_config.repetition_penalty != null && generation_config.repetition_penalty != 1.0) {
      processors.add(RepetitionPenaltyLogitsProcessor(generation_config.repetition_penalty));
    }

    if (generation_config.no_repeat_ngram_size != null && generation_config.no_repeat_ngram_size > 0) {
      processors.add(NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size));
    }

    // Below was commented out in transformers.js:
    // if (generation_config.encoder_no_repeat_ngram_size !== null && generation_config.encoder_no_repeat_ngram_size > 0) {
    //     if (this.config.is_encoder_decoder) {
    //         processors.push(new EncoderNoRepeatNGramLogitsProcessor(
    //             generation_config.encoder_no_repeat_ngram_size,
    //             encoder_input_ids
    //         ));
    //     } else {
    //         throw new Error("It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture");
    //     }
    // }

    if (generation_config.bad_words_ids != null) {
      processors.add(NoBadWordsLogitsProcessor(generation_config.bad_words_ids!, generation_config.eos_token_id));
    }

    if (generation_config.min_length != null && generation_config.eos_token_id != null && generation_config.min_length > 0) {
      processors.add(MinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id));
    }

    if (generation_config.min_new_tokens != null && generation_config.eos_token_id != null && generation_config.min_new_tokens! > 0) {
      processors.add(MinNewTokensLengthLogitsProcessor(
          input_ids_seq_length,
          generation_config.min_new_tokens!,
          generation_config.eos_token_id
      ));
    }

    // Below was commented out in transformers.js:
    // if (prefix_allowed_tokens_fn !== null) {
    //     processors.push(new PrefixConstrainedLogitsProcessor(
    //         prefix_allowed_tokens_fn,
    //         generation_config.num_beams / generation_config.num_beam_groups
    //     ));
    // }


    if (generation_config.forced_bos_token_id != null) {
      processors.add(ForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id!));
    }

    if (generation_config.forced_eos_token_id != null) {
      processors.add(ForcedEOSTokenLogitsProcessor(
          generation_config.max_length,
          generation_config.forced_eos_token_id
      ));
    }

    // Below was commented out in transformers.js:
    // if (generation_config.remove_invalid_values === true) {
    //     processors.push(new InfNanRemoveLogitsProcessor());
    // }

    // Below was commented out in transformers.js:
    // if (generation_config.exponential_decay_length_penalty !== null) {
    //     processors.push(new ExponentialDecayLengthPenalty(
    //         generation_config.exponential_decay_length_penalty,
    //         generation_config.eos_token_id,
    //         input_ids_seq_length
    //     ));
    // }

    // Below was commented out in transformers.js:
    // if (generation_config.suppress_tokens !== null) {
    //     processors.push(new SuppressTokensLogitsProcessor(generation_config.suppress_tokens));
    // }

    if (generation_config.begin_suppress_tokens != null) {
      final begin_index = (input_ids_seq_length > 1 || generation_config.forced_bos_token_id == null)
          ? input_ids_seq_length
          : input_ids_seq_length + 1;

      processors.add(SuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens!, begin_index));
    }

    // Below was commented out in transformers.js:
    // DEPRECATED: https://github.com/huggingface/transformers/pull/29485
    // if (generation_config.forced_decoder_ids !== null) {
    //     processors.push(new ForceTokensLogitsProcessor(generation_config.forced_decoder_ids));
    // }


    // 8. prepare batched CFG externally
    if (generation_config.guidance_scale != null && generation_config.guidance_scale! > 1) {
      processors.add(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale!));
    }

    if (logits_processor != null) {
      processors.extend(logits_processor);
    }

    // Below was commented out in transformers.js:
    // `LogitNormalization` should always be the last logit processor, when present
    // if (generation_config.renormalize_logits === true) {
    //     processors.push(new LogitNormalization());
    // }

    return processors;
  }

  /// This function merges multiple generation configs together to form a final generation config to be used by the model for text generation.
  /// It first creates an empty `GenerationConfig` object, then it applies the model's own `generation_config` property to it. Finally, if a `generation_config` object was passed in the arguments, it overwrites the corresponding properties in the final config with those of the passed config object.
  /// @param {GenerationConfig|null} generation_config A `GenerationConfig` object containing generation parameters.
  /// @param {Object} kwargs Additional generation parameters to be used in place of those in the `generation_config` object.
  /// @returns {GenerationConfig} The final generation config object to be used by the model for text generation.
  GenerationConfig _prepare_generation_config(
    GenerationConfig? generation_config,
    Map<String, dynamic>? kwargs,
    [GenerationConfig Function(Map<String, dynamic> json)? cls]
  ) {
    cls ??= GenerationConfig.fromJson;

    // Create empty generation config (contains defaults)
    // We pass `this.config` so that if `eos_token_id` or `bos_token_id` exist in the model's config, we will use them
    final Map<String, dynamic> config = { ...?this.config?.toJson() };
    for (final key in ['decoder', 'generator', 'text_config']) {
      // Special case: some models have generation attributes set in the decoder.
      // Use them if still unset in the generation config.
      if (config.containsKey(key)) {
        config.addAll(config[key]..removeWhere((_, v) => v != null));
      }
    }

    final gen_config = cls(config).toJson();

    // Apply model's generation config, if it exists
    gen_config.addAll((this.generation_config?.toJson() ?? {})..removeWhere((_, v) => v == null));

    // Next, use any generation config specified by the user
    // when calling `generate`
    if (generation_config != null) {
      gen_config.addAll(generation_config.toJson()..removeWhere((_, v) => v == null));
    }

    // Finally, if any kwargs were passed, use them to overwrite
    if (kwargs != null) {
      // Using gen_config.keys as a replacement for Object.getOwnPropertyNames
      // should work since gen_config should have all the properties of the
      // class set which means toJson() should contain all the keys as the
      // property names.
      gen_config.addAll(pick(
        kwargs,
        cls({}).toJson().keys.toList(),
      )..removeWhere((_, v) => v == null));
    }

    return cls(gen_config);
  }

  ///
  /// @param {GenerationConfig} generation_config
  /// @param {StoppingCriteriaList} [stopping_criteria=null]
  StoppingCriteriaList _get_stopping_criteria(
    GenerationConfig generation_config,
    [StoppingCriteriaList? stopping_criteria]
  ) {
    final criteria = StoppingCriteriaList();

    if (generation_config.max_length != null) {
      criteria.push(MaxLengthCriteria(
        generation_config.max_length,
        config?.max_position_embeddings,
      ));
    }
    // Below was commented out in transformers.js:
    // if (generation_config.max_time !== null) {
    //     criteria.push(new MaxTimeCriteria(generation_config.max_time));
    // }
    if (generation_config.eos_token_id != null) {
      criteria.push(EosTokenCriteria(generation_config.eos_token_id));
    }

    if (stopping_criteria != null) {
      criteria.extend(stopping_criteria);
    }

    return criteria;
  }

  /// Confirms that the model class is compatible with generation.
  /// If not, raises an exception that points to the right class to use.
  void _validate_model_class() {
    if (!can_generate) {
      final generate_compatible_mappings = [
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        // MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING, // TODO
        MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
        MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
      ];

      final modelName = MODEL_CLASS_TO_NAME_MAPPING[reflection.type];

      final Set<dynamic> generate_compatible_classes = {};
      final modelType = config?.model_type;
      for (final model_mapping in generate_compatible_mappings) {
        final supported_models = model_mapping[modelType];
        if (supported_models != null) {
          generate_compatible_classes.add(supported_models.$1);
        }
      }

      String errorMessage = "The current model class ($modelName) is not compatible with `.generate()`, as it doesn't have a language model head.";
      if (generate_compatible_classes.isNotEmpty) {
        errorMessage += ' Please use the following class instead: ${[...generate_compatible_classes].join(', ')}';
      }
      throw Exception(errorMessage);
    }
  }

  Future<Map<String, dynamic>> prepare_inputs_for_generation(
    List<List<int>> input_ids,
    Map<String, dynamic> model_inputs,
    dynamic generation_config,
  ) async {
    if (_prepare_inputs_for_generation == null) {
      throw Exception('`_prepare_inputs_for_generation` is not initialized');
    }

    return await _prepare_inputs_for_generation!(input_ids, model_inputs, generation_config);
  }

  /// @param {Object} inputs
  /// @param {bigint[][]} inputs.generated_input_ids
  /// @param {Object} inputs.outputs
  /// @param {Object} inputs.model_inputs
  /// @param {boolean} inputs.is_encoder_decoder
  /// @returns {Object} The updated model inputs for the next generation iteration.
  Future<Map<String, dynamic>> _update_model_kwargs_for_generation({
    required List<List<int>> generated_input_ids,
    required Map<String, dynamic> outputs,
    required Map<String, dynamic> model_inputs,
    required bool is_encoder_decoder,
  }) async {
    // update past_key_values
    model_inputs['past_key_values'] = getPastKeyValues(outputs, model_inputs['past_key_values']);

    // update inputs for next run
    model_inputs['input_ids'] = await Tensor.create(
      TensorDataType.int64,
      generated_input_ids.expand((e) => e).toList(),
      [generated_input_ids.length, 1],
    );

    if (!is_encoder_decoder) {
      // update attention mask
      model_inputs['attention_mask'] = await cat([
        model_inputs['attention_mask'] as Tensor,
        await ones([(model_inputs['attention_mask'] as Tensor).dims[0], 1]),
      ], 1);
    } else if (model_inputs.containsKey('decoder_attention_mask')) {
      // TODO: update decoder attention mask if the model requires it
    }

    // force recreate position_ids in next iteration
    model_inputs['position_ids'] = null;

    return model_inputs;
  }

  /// This function extracts the model-specific `inputs` for generation.
  /// @param {Object} params
  /// @param {Tensor} [params.inputs=null]
  /// @param {number} [params.bos_token_id=null]
  /// @param {Record<string, Tensor|number[]>} [params.model_kwargs]
  /// @returns {{inputs_tensor: Tensor, model_inputs: Record<string, Tensor>, model_input_name: string}} The model-specific inputs for generation.
  ({
    Tensor? inputs_tensor,
    Map<String, dynamic> model_inputs,
    String input_name,
  }) _prepare_model_inputs({
    Tensor? inputs,
    int? bos_token_id,
    Map<String, dynamic> model_kwargs = const {},
  }) {
    final Map<String, dynamic> model_inputs = pick(model_kwargs, forward_params);
    final input_name = main_input_name;
    if (model_inputs.containsKey(input_name)) {
      if (inputs != null) {
        throw Exception(
            '`inputs`: {inputs}` were passed alongside {input_name} which is not allowed. '
                'Make sure to either pass {inputs} or {input_name}=...'
        );
      }
    } else {
      if (inputs != null) {
        model_inputs[input_name] = inputs;
      }
    }

    final inputs_tensor = model_inputs[input_name];

    return (
      inputs_tensor: inputs_tensor,
      model_inputs: model_inputs,
      input_name: input_name,
    );
  }

  /// TODO: Add types
  _prepare_inputs_embeds(Map<String, dynamic> args) async {
    // { input_ids, pixel_values, inputs_embeds, attention_mask }
    throw UnimplementedError('`_prepare_inputs_embeds` should be implemented in a subclass');
  }

  Future<Map<String, dynamic>> _prepare_encoder_decoder_kwargs_for_generation(Map<String, dynamic> kwargs) async {
    dynamic inputs_tensor = kwargs['inputs_tensor'];
    Map<String, dynamic> model_inputs = kwargs['model_inputs'];
    dynamic model_input_name = kwargs['model_input_name'];
    dynamic generation_config = kwargs['generation_config'];

    if (sessions['model']!.inputNames.contains('inputs_embeds')
        && model_inputs['inputs_embeds'] != null
        && _is_prepare_inputs_embeds_supported) {
      // Encoder expects `inputs_embeds` instead of `input_ids`
      model_inputs = { ...model_inputs };
      final input_ids = model_inputs.remove('input_ids'),
          pixel_values = model_inputs.remove('pixel_values'),
          attention_mask = model_inputs.remove('attention_mask'),
          kwargs = model_inputs;
      final prepared_inputs = await _prepare_inputs_embeds(model_inputs);
      model_inputs = {
        ...kwargs,
        ...pick(prepared_inputs, ['inputs_embeds', 'attention_mask']),
      };
    }

    final encoderForwardOutput = await encoderForward(model_inputs);
    Tensor last_hidden_state = encoderForwardOutput['last_hidden_state']!;

    // for classifier free guidance we need to add a 'null' input to our encoder hidden states
    if (generation_config['guidance_scale'] != null && generation_config['guidance_scale'] > 1) {
      last_hidden_state = await cat([
        last_hidden_state,
        await full_like(last_hidden_state, 0.0),
      ], 0);

      if (model_inputs.containsKey('attention_mask')) {
        model_inputs['attention_mask'] = await cat([
          model_inputs['attention_mask'] as Tensor,
          await zeros_like(model_inputs['attention_mask']),
        ], 0);
      }
    } else if (model_inputs['decoder_input_ids'] != null) {
      // Ensure that the encoder outputs have the same batch size as the decoder inputs,
      // allowing for more efficient batched generation for single inputs
      final decoder_input_ids_batch_size = (await toI64Tensor(model_inputs['decoder_input_ids'])).dims[0];
      if (decoder_input_ids_batch_size != last_hidden_state.dims[0]) {
        if (last_hidden_state.dims[0] != 1) {
          throw Exception(
              'The encoder outputs have a different batch size (${last_hidden_state.dims[0]}) than the decoder inputs ($decoder_input_ids_batch_size).'
          );
        }
        last_hidden_state = await cat(List<Tensor>.generate(decoder_input_ids_batch_size, (_) => last_hidden_state), 0);
      }
    }
    model_inputs['encoder_outputs'] = last_hidden_state;

    return model_inputs;
  }

  /// Prepares `decoder_input_ids` for generation with encoder-decoder models
  /// @param {*} param0
  Future<Map<String, dynamic>> _prepare_decoder_input_ids_for_generation(Map<String, dynamic> kwargs) async {
    final int batch_size = kwargs['batch_size'];
    final model_input_name = kwargs['model_input_name'];
    final Map<String, dynamic> model_kwargs = kwargs['model_kwargs'];
    dynamic decoder_start_token_id = kwargs['decoder_start_token_id'];
    int? bos_token_id = kwargs['bos_token_id'];
    final generation_config = kwargs['generation_config'];

    dynamic decoder_input_ids = model_kwargs.remove('decoder_input_ids');
    final model_inputs = model_kwargs;

    // Prepare input ids if the user has not defined `decoder_input_ids` manually.
    if (decoder_input_ids is! Tensor) {
      if (decoder_input_ids != null) {
        decoder_start_token_id ??= bos_token_id;

        if (config?.model_type == 'musicgen') {
          // Custom logic (TODO: move to Musicgen class)
          decoder_input_ids = List<List<int>>.generate(
            batch_size * config!['decoder']['num_codebooks'] as int,
                (_) => [decoder_start_token_id],
          );
        } else if (decoder_start_token_id is List) {
          if (decoder_start_token_id.length != batch_size) {
            throw Exception(
                '`decoder_start_token_id` expected to have length $batch_size but got ${decoder_start_token_id.length}'
            );
          }
          decoder_input_ids = decoder_start_token_id;
        } else {
          decoder_input_ids = List<List<int>>.generate(
            batch_size,
                (_) => [decoder_start_token_id],
          );
        }
      } else if (decoder_input_ids[0] is! List) {
        // Correct batch size
        decoder_input_ids = List<List<int>>.generate(
          batch_size,
              (_) => decoder_input_ids,
        );
      }
      decoder_input_ids = await toI64Tensor(decoder_input_ids);
    }

    model_kwargs['decoder_attention_mask'] = await ones_like(decoder_input_ids);

    return { 'input_ids': decoder_input_ids, 'model_inputs': model_inputs };
  }

  /// Generates sequences of token ids for models with a language modeling head.
  /// @param {import('./generation/parameters.js').GenerationFunctionParameters} options
  /// @returns {Promise<ModelOutput|Tensor>} The output of the model, which can contain the generated token ids, attentions, and scores.
  Future<dynamic> generate(Map<String, dynamic> kwargs) async {
    kwargs = { ...kwargs };

    final inputs = kwargs.remove('inputs');
    GenerationConfig? generation_config = kwargs.remove('generation_config');
    final logits_processor = kwargs.remove('logits_processor');
    final stopping_criteria = kwargs.remove('stopping_criteria');
    final streamer = kwargs.remove('streamer');

    _validate_model_class();

    // Update generation config with defaults and kwargs
    generation_config = _prepare_generation_config(generation_config, kwargs);

    // 3. Define model inputs
    final preparedModelInputs = _prepare_model_inputs(
      inputs: inputs,
      model_kwargs: kwargs,
    );
    final inputs_tensor = preparedModelInputs.inputs_tensor;
    Map<String, dynamic> model_inputs = preparedModelInputs.model_inputs;
    final model_input_name = preparedModelInputs.input_name;

    final is_encoder_decoder = config?.is_encoder_decoder;

    // 4. Define other model kwargs
    if (!(is_encoder_decoder ?? false)) {
      // decoder-only models should use left-padding for generation
    } else if (!(model_inputs.containsKey('encoder_outputs'))) {
      // if model is encoder decoder encoder_outputs are created
      // and added to `model_kwargs`
      model_inputs = await _prepare_encoder_decoder_kwargs_for_generation({
        'inputs_tensor': inputs_tensor,
        'model_inputs': model_inputs,
        'model_input_name': model_input_name,
        'generation_config': generation_config,
      });
    }

    // 5. Prepare `input_ids` which will be used for auto-regressive generation
    // TODO: Update to align with HF transformers' implementation
    Tensor input_ids;
    if (is_encoder_decoder ?? false) {
      // Generating from the encoder outputs
      final preparedDecoderInputIds = await _prepare_decoder_input_ids_for_generation({
        'batch_size': (model_inputs[model_input_name] as Tensor?)?.dims.first,
        'model_input_name': model_input_name,
        'model_kwargs': model_inputs,
        'decoder_start_token_id': generation_config.decoder_start_token_id,
        'bos_token_id': generation_config.bos_token_id,
        'generation_config': generation_config,
      });
      input_ids = preparedDecoderInputIds['input_ids'];
      model_inputs = preparedDecoderInputIds['model_inputs'];
    } else {
      input_ids = model_inputs[model_input_name]!;
    }

    // 6. Prepare `max_length` depending on other stopping criteria.
    int input_ids_length = input_ids.dims.last;

    if (generation_config.max_new_tokens != null) {
      generation_config.max_length = input_ids_length + generation_config.max_new_tokens!;
    }

    // Below was commented out in transformers.js:
    // input_ids_length = model_inputs[model_input_name].dims.at(1);
    // // inputs instanceof Tensor ?  : inputs.length;

    // Below was commented out in transformers.js:
    // // decoder-only
    // if (input_ids_length === 0) {
    //     throw Error("Must supply a non-empty array of input token ids.")
    // }

    // Below was commented out in transformers.js:
    // let decoder_input_ids =
    // generation_config.decoder_input_ids
    // ?? generation_config.decoder_start_token_id
    // ?? generation_config.bos_token_id
    // ?? generation_config.eos_token_id;

    // Update logits processor
    // 8. prepare distribution pre_processing samplers
    final prepared_logits_processor = _get_logits_processor(
      generation_config,
      input_ids_length,
      logits_processor,
    );

    // 9. prepare stopping criteria
    final prepared_stopping_criteria = _get_stopping_criteria(
      generation_config,
      stopping_criteria,
    );

    // Below was commented out in transformers.js:
    // /** @type {number[]} */
    // let eos_token_ids = generation_config.eos_token_id;
    // if (eos_token_ids !== null && !Array.isArray(eos_token_ids)) {
    //     eos_token_ids = [eos_token_ids];
    // }

    final numInputs = (model_inputs[model_input_name]! as Tensor).dims.first;

    // Below was commented out in transformers.js:
    // TODO:
    // done is a list of booleans to keep track of which inputs are done
    // const done = new Array(numInputs).fill(false);
    // For efficiency purposes, we remove completed rows from model_inputs
    // when the beam is complete, and we keep track of the row index
    // const rowIndexToBatchIndex = new Map();

    final LogitsSampler sampler = LogitsSampler.getSampler(generation_config);

    // TODO make > numInputs
    final scores = List<double>.filled(numInputs, 0);
    final List<List<int>> all_input_ids = input_ids.tolist()
        .map((x) => List<int>.from(x)).toList();
    if (streamer != null) {
      streamer.put(all_input_ids);
    }
    // Below was commented out in transformers.js:
    // const all_generated_input_ids = Array.from({ length: numInputs }, () => []);

    // Below was commented out in transformers.js:
    // NOTE: For now, we don't support spawning new beams
    // TODO: when we do, we simply copy past key values and accumulate into single large tensor

    ////////////////////////////////////////////////////
    // Generic search which handles 4 generation modes:
    // - GenerationMode.GREEDY_SEARCH
    // - GenerationMode.SAMPLE
    // - GenerationMode.BEAM_SEARCH
    // - GenerationMode.BEAM_SAMPLE
    ////////////////////////////////////////////////////
    Map<String, Tensor<dynamic>> outputs;
    Map<String, dynamic> attentions = {};
    while (true) {
      // prepare model inputs
      model_inputs = await prepare_inputs_for_generation(
        all_input_ids,
        model_inputs,
        generation_config,
      );
      outputs = await forward(model_inputs);

      if (generation_config.output_attentions && generation_config.return_dict_in_generate) {
        // Get attentions if they are present
        final token_attentions = getAttentions(outputs);
        for (final key in token_attentions.keys) {
          if (!attentions.containsKey(key)) {
            attentions[key] = [];
          }
          attentions[key].push(token_attentions[key]);
        }
      }

      // Logits are of the form [batch_size, out_seq_length, vocab_size]
      // In most cases, this will be [batch_size, 1, vocab_size]
      // So, we select the last token's logits:
      // (equivalent to `logits = outputs.logits[:, -1, :]`)
      final logits = await outputs['logits']!.slice([null, -1, null]);

      final next_tokens_scores = await prepared_logits_processor(all_input_ids, logits);

      final List<List<int>> generated_input_ids = [];
      // const new_kv_cache = [];// NOTE: Only used for beam search when concatenating new kv
      // Loop over each batch
      for (int batch_idx = 0; batch_idx < next_tokens_scores.dims.first; ++batch_idx) {
        final logs = await next_tokens_scores[batch_idx];

        final sampledTokens = await sampler(logs);
        for (final (newTokenId, logProb) in sampledTokens) {
          // TODO: If branching, use previous beam as a starting point
          // update generated ids, model inputs, and length for next step
          scores[batch_idx] += logProb;
          all_input_ids[batch_idx].add(newTokenId);
          generated_input_ids.add([newTokenId]);

          // TODO: Support beam search
          break;
        }
      }
      if (streamer != null) {
        streamer.put(generated_input_ids);
      }

      final stop = prepared_stopping_criteria(all_input_ids);
      if (stop.every((x) => x)) {
        break;
      }

      model_inputs = await _update_model_kwargs_for_generation(
        generated_input_ids: generated_input_ids,
        outputs: outputs,
        model_inputs: model_inputs,
        is_encoder_decoder: is_encoder_decoder ?? false,
      );
    }

    if (streamer != null) {
      streamer.end();
    }

    // Retrieve and dispose all final past key values (including encoder attentions)
    final past_key_values = getPastKeyValues(
      outputs,
      model_inputs['past_key_values'],
      true,
    );

    // TODO: ensure all_input_ids is padded correctly...
    final sequences = await Tensor.create(
      TensorDataType.int64,
      all_input_ids.flat(),
      [all_input_ids.length, all_input_ids[0].length],
    );

    if (generation_config.return_dict_in_generate) {
      return <String, dynamic>{
        'sequences': sequences,
        'past_key_values': past_key_values,
        ...attentions,
        // TODO:
        // scores,
        // logits,
      };
    } else {
      // Dispose all remaining tensors
      for (final tensor in outputs.values) {
        if (tensor.location == 'gpu-buffer') {
          await tensor.dispose();
        }
      }
      return sequences;
    }
  }

  /// Returns an object containing past key values from the given decoder results object.
  ///
  /// @param {Object} decoderResults The decoder results object.
  /// @param {Object} pastKeyValues The previous past key values.
  /// @returns {Object} An object containing past key values.
  Map<String, dynamic> getPastKeyValues(
      Map<String, dynamic> decoderResults,
      [Map<String, dynamic>? pastKeyValues, bool disposeEncoderPKVs = false]
      ) {
    final Map<String, dynamic> pkvs = {};

    for (final name in decoderResults.keys) {
      if (name.startsWith('present')) {
        final newName = name.replaceAll('present', 'past_key_values');
        final is_encoder_pkv = name.contains('encoder');
        if (is_encoder_pkv && pastKeyValues != null) {
          // Optimization introduced by optimum to reuse past key values.
          // So, we just replace the constant outputs (`decoderResults[name]`) with the previous past key values.
          // https://github.com/huggingface/optimum/blob/0bf2c05fb7e1182b52d21b703cfc95fd9e4ea3dc/optimum/onnxruntime/base.py#L677-L704
          pkvs[newName] = pastKeyValues[newName];
        } else { // decoder or using first encoder PKVs
          pkvs[newName] = decoderResults[name];
        }

        if (pastKeyValues != null && (!is_encoder_pkv || disposeEncoderPKVs)) {
          // - Always dispose decoder PKVs
          // - Only dispose encoder past key values when requested (after generation)
          final t = pastKeyValues[newName];
          if (t.location == 'gpu-buffer') {
            t.dispose();
          }
        }
      }
    }
    return pkvs;
  }

  /// Returns an object containing attentions from the given model output object.
  ///
  /// @param {Object} model_output The output of the model.
  /// @returns {{cross_attentions?: Tensor[]}} An object containing attentions.
  Map<String, dynamic> getAttentions(Map<String, dynamic> model_output) {
    final Map<String, dynamic> attentions = {};

    for (final attnName in ['cross_attentions', 'encoder_attentions', 'decoder_attentions']) {
      for (final name in model_output.keys) {
        if (name.startsWith(attnName)) {
          if (!(attentions.containsKey(attnName))) {
            attentions[attnName] = [];
          }
          attentions[attnName].add(model_output[name]);
        }
      }
    }
    return attentions;
  }

  /// Adds past key values to the decoder feeds object. If pastKeyValues is null, creates new tensors for past key values.
  ///
  /// @param {Object} decoderFeeds The decoder feeds object to add past key values to.
  /// @param {Object} pastKeyValues An object containing past key values.
  @override
  Future<void> addPastKeyValues(Map<String, dynamic> decoderFeeds, Map<String, dynamic>? pastKeyValues) async {
    if (pastKeyValues != null) {
      decoderFeeds.addAll(pastKeyValues);
    } else {
      // TODO: Unfortunately, flutter_onnxruntime doesn't actually expose this config we add it on using extensions
      final session = sessions['decoder_model_merged'] ?? sessions['model'];
      final dtype = session?.config['kv_cache_dtype'] ?? 'float32';
      final empty = [];

      final batch_size = (
          (decoderFeeds[main_input_name] ?? decoderFeeds['attention_mask']) as Tensor?
      )?.dims[0] ?? 1;
      final shapes = getKeyValueShapes(config, batch_size: batch_size);

      for (final name in shapes.keys) {
        decoderFeeds[name] = await Tensor.create(dtype, empty, shapes[name]);
      }
    }
  }

  @override
  Future<Tensor> encode_image(Map<String, dynamic> params) async {
    return await super.encode_image(params);
  }

  @override
  Future<Tensor> encode_text(Map<String, dynamic> params) async {
    return await super.encode_text(params);
  }

  @override
  Future<Tensor> encode_audio(Map<String, dynamic> params) async {
    return await super.encode_audio(params);
  }
}

//////////////////////////////////////////////////
// Base model output class
class ModelOutput {}

//////////////////////////////////////////////////
// Bert models
class BertPreTrainedModel extends PreTrainedModel {
  BertPreTrainedModel(super.reflection, super.config, super.sessions, super.configs);
}
class BertModel extends BertPreTrainedModel {
  @override
  final PreTrainedModelReflection reflection = _reflection;

  static final PreTrainedModelReflection _reflection = PreTrainedModelReflection(
    type: BertModel,
    constructor: constructor,
  );

  static PreTrainedModel constructor(
    PretrainedConfig? config,
    Map<String, OrtSession> sessions,
    Map<String, Map<String, dynamic>> configs,
  ) => BertModel(_reflection, config, sessions, configs);

  static final ModelClassInfo modelClassInfo = ModelClassInfo(
    BertModel,
    PreTrainedModel.setup_from_pretrained(_reflection),
  );

  BertModel(super.reflection, super.config, super.sessions, super.configs);
}

/// BertForMaskedLM is a class representing a BERT model for masked language modeling.
class BertForMaskedLM extends BertPreTrainedModel {
  @override
  final PreTrainedModelReflection reflection = _reflection;

  static final PreTrainedModelReflection _reflection = PreTrainedModelReflection(
    type: BertForMaskedLM,
    constructor: constructor,
  );

  static PreTrainedModel constructor(
    PretrainedConfig? config,
    Map<String, OrtSession> sessions,
    Map<String, Map<String, dynamic>> configs,
  ) => BertForMaskedLM(_reflection, config, sessions, configs);

  static final ModelClassInfo modelClassInfo = ModelClassInfo(
    BertForMaskedLM,
    PreTrainedModel.setup_from_pretrained(_reflection),
  );

  BertForMaskedLM(super.reflection, super.config, super.sessions, super.configs);

  /// Calls the model on new inputs.
  ///
  /// @param {Object} model_inputs The inputs to the model.
  /// @returns {Promise<MaskedLMOutput>} An object containing the model's output logits for masked language modeling.
  /// TODO: Because [PreTrainedModel]'s [call] return type is [Map<String, Tensor>] we can't use [MaskedLMOutput]. Maybe we should update this?
  @override
  Future<Map<String, Tensor>> call(Map<String, dynamic> model_inputs) async {
    return { 'logits': MaskedLMOutput(await super.call(model_inputs)).logits };
  }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// XLMRoberta models
class XLMRobertaPreTrainedModel extends PreTrainedModel {
  XLMRobertaPreTrainedModel(super.reflection, super.config, super.sessions, super.configs);
}
class XLMRobertaModel extends XLMRobertaPreTrainedModel {
  @override
  final PreTrainedModelReflection reflection = _reflection;

  static final PreTrainedModelReflection _reflection = PreTrainedModelReflection(
    type: XLMRobertaModel,
    constructor: constructor,
  );

  static PreTrainedModel constructor(
    PretrainedConfig? config,
    Map<String, OrtSession> sessions,
    Map<String, Map<String, dynamic>> configs,
  ) => XLMRobertaModel(_reflection, config, sessions, configs);

  static final ModelClassInfo modelClassInfo = ModelClassInfo(
    XLMRobertaModel,
    PreTrainedModel.setup_from_pretrained(_reflection),
  );

  XLMRobertaModel(super.reflection, super.config, super.sessions, super.configs);
}

/// XLMRobertaForMaskedLM class for performing masked language modeling on XLMRoberta models.
class XLMRobertaForMaskedLM extends XLMRobertaPreTrainedModel {
  @override
  final PreTrainedModelReflection reflection = _reflection;

  static final PreTrainedModelReflection _reflection = PreTrainedModelReflection(
    type: XLMRobertaForMaskedLM,
    constructor: constructor,
  );

  static PreTrainedModel constructor(
    PretrainedConfig? config,
    Map<String, OrtSession> sessions,
    Map<String, Map<String, dynamic>> configs,
  ) => XLMRobertaForMaskedLM(_reflection, config, sessions, configs);

  static final ModelClassInfo modelClassInfo = ModelClassInfo(
    XLMRobertaForMaskedLM,
    PreTrainedModel.setup_from_pretrained(_reflection),
  );

  XLMRobertaForMaskedLM(super.reflection, super.config, super.sessions, super.configs);

  /// Calls the model on new inputs.
  ///
  /// @param {Object} model_inputs The inputs to the model.
  /// @returns {Promise<MaskedLMOutput>} returned object
  /// TODO: Because [PreTrainedModel]'s [call] return type is [Map<String, Tensor>] we can't use [MaskedLMOutput]. Maybe we should update this?
  @override
  Future<Map<String, Tensor>> call(Map<String, dynamic> model_inputs) async {
    return { 'logits': MaskedLMOutput(await super.call(model_inputs)).logits };
  }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Idefics3 Models
class Idefics3PreTrainedModel extends PreTrainedModel {
  @override
  List<String> forward_params = [
    'input_ids',
    'attention_mask',
    'pixel_values',
    'pixel_attention_mask',
    'position_ids',
    'past_key_values',
  ];

  Idefics3PreTrainedModel(super.reflection, super.config, super.sessions, super.configs);
}

/// The Idefics3 model which consists of a vision backbone and a language model.
class Idefics3ForConditionalGeneration extends Idefics3PreTrainedModel {
  @override
  final PreTrainedModelReflection reflection = _reflection;

  static final PreTrainedModelReflection _reflection = PreTrainedModelReflection(
    type: Idefics3ForConditionalGeneration,
    constructor: constructor,
  );

  static PreTrainedModel constructor(
    PretrainedConfig? config,
    Map<String, OrtSession> sessions,
    Map<String, Map<String, dynamic>> configs,
  ) => Idefics3ForConditionalGeneration(_reflection, config, sessions, configs);

  static final ModelClassInfo modelClassInfo = ModelClassInfo(
    Idefics3ForConditionalGeneration,
    PreTrainedModel.setup_from_pretrained(_reflection),
  );

  Idefics3ForConditionalGeneration(super.reflection, super.config, super.sessions, super.configs);

  @override
  Future<Tensor> encode_image(Map<String, dynamic> params) async {
    final features = (await sessionRun(
      sessions['vision_encoder']!,
      {
        'pixel_values': params['pixel_values'],
        'pixel_attention_mask': params['pixel_attention_mask'],
      },
    ))['image_features']!;
    return features;
  }

  @override
  Future<({Tensor inputs_embeds, dynamic attention_mask})> _merge_input_ids_with_image_features(Map<String, dynamic> kwargs) async {
    final image_features = kwargs['image_features'] as Tensor;
    final vision_hidden_size = image_features.dims.last;
    final reshaped_image_hidden_states = await image_features.view([-1, vision_hidden_size]);

    return await default_merge_input_ids_with_image_features({
      'image_token_id': config?['image_token_id'],
      ...kwargs,
      'image_features': reshaped_image_hidden_states,
    });
  }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// GPT2 models
class GPT2PreTrainedModel extends PreTrainedModel {
  GPT2PreTrainedModel(super.reflection, super.config, super.sessions, super.configs);
}

class GPT2Model extends GPT2PreTrainedModel {
  GPT2Model(super.reflection, super.config, super.sessions, super.configs);
}

/// GPT-2 language model head on top of the GPT-2 base model. This model is suitable for text generation tasks.
class GPT2LMHeadModel extends GPT2PreTrainedModel {
  @override
  final PreTrainedModelReflection reflection = _reflection;

  static final PreTrainedModelReflection _reflection = PreTrainedModelReflection(
    type: GPT2LMHeadModel,
    constructor: constructor,
  );

  static PreTrainedModel constructor(
    PretrainedConfig? config,
    Map<String, OrtSession> sessions,
    Map<String, Map<String, dynamic>> configs,
  ) => GPT2LMHeadModel(_reflection, config, sessions, configs);

  static final ModelClassInfo modelClassInfo = ModelClassInfo(
    GPT2LMHeadModel,
    PreTrainedModel.setup_from_pretrained(_reflection),
  );

  GPT2LMHeadModel(super.reflection, super.config, super.sessions, super.configs);
}
// export class GPT2ForSequenceClassification extends GPT2PreTrainedModel {
// TODO
// }
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// GraniteMoeHybrid models
class GraniteMoeHybridPreTrainedModel extends PreTrainedModel {
  GraniteMoeHybridPreTrainedModel(super.reflection, super.config, super.sessions, super.configs);
}
class GraniteMoeHybridModel extends GraniteMoeHybridPreTrainedModel {
  @override
  final PreTrainedModelReflection reflection = _reflection;

  static final PreTrainedModelReflection _reflection = PreTrainedModelReflection(
    type: GraniteMoeHybridModel,
    constructor: constructor,
  );

  static PreTrainedModel constructor(
    PretrainedConfig? config,
    Map<String, OrtSession> sessions,
    Map<String, Map<String, dynamic>> configs,
  ) => GraniteMoeHybridModel(_reflection, config, sessions, configs);

  static final ModelClassInfo modelClassInfo = ModelClassInfo(
    GraniteMoeHybridModel,
    PreTrainedModel.setup_from_pretrained(_reflection),
  );

  GraniteMoeHybridModel(super.reflection, super.config, super.sessions, super.configs);
}
class GraniteMoeHybridForCausalLM extends GraniteMoeHybridPreTrainedModel {
  @override
  final PreTrainedModelReflection reflection = _reflection;

  static final PreTrainedModelReflection _reflection = PreTrainedModelReflection(
    type: GraniteMoeHybridForCausalLM,
    constructor: constructor,
  );

  static PreTrainedModel constructor(
    PretrainedConfig? config,
    Map<String, OrtSession> sessions,
    Map<String, Map<String, dynamic>> configs,
  ) => GraniteMoeHybridForCausalLM(_reflection, config, sessions, configs);

  static final ModelClassInfo modelClassInfo = ModelClassInfo(
    GraniteMoeHybridForCausalLM,
    PreTrainedModel.setup_from_pretrained(_reflection),
  );

  GraniteMoeHybridForCausalLM(super.reflection, super.config, super.sessions, super.configs);
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Qwen3 models

/// The bare Qwen3 Model outputting raw hidden-states without any specific head on top.
class Qwen3PreTrainedModel extends PreTrainedModel {
  Qwen3PreTrainedModel(super.reflection, super.config, super.sessions, super.configs);
}

/// The bare Qwen3 Model outputting raw hidden-states without any specific head on top.
class Qwen3Model extends Qwen3PreTrainedModel {
  @override
  final PreTrainedModelReflection reflection = _reflection;

  static final PreTrainedModelReflection _reflection = PreTrainedModelReflection(
    type: Qwen3Model,
    constructor: constructor,
  );

  static PreTrainedModel constructor(
    PretrainedConfig? config,
    Map<String, OrtSession> sessions,
    Map<String, Map<String, dynamic>> configs,
  ) => Qwen3Model(_reflection, config, sessions, configs);

  static final ModelClassInfo modelClassInfo = ModelClassInfo(
    Qwen3Model,
    PreTrainedModel.setup_from_pretrained(_reflection),
  );

  Qwen3Model(super.reflection, super.config, super.sessions, super.configs);
}

class Qwen3ForCausalLM extends Qwen3PreTrainedModel {
  @override
  final PreTrainedModelReflection reflection = _reflection;

  static final PreTrainedModelReflection _reflection = PreTrainedModelReflection(
    type: Qwen3ForCausalLM,
    constructor: constructor,
  );

  static PreTrainedModel constructor(
    PretrainedConfig? config,
    Map<String, OrtSession> sessions,
    Map<String, Map<String, dynamic>> configs,
  ) => Qwen3ForCausalLM(_reflection, config, sessions, configs);

  static final ModelClassInfo modelClassInfo = ModelClassInfo(
    Qwen3ForCausalLM,
    PreTrainedModel.setup_from_pretrained(_reflection),
  );

  Qwen3ForCausalLM(super.reflection, super.config, super.sessions, super.configs);
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// AutoModels, used to simplify construction of PreTrainedModels
// (uses config to instantiate correct class)

// abstract class PretrainedBase {
//   /// Mapping from model type to model class.
//   /// @type {Map<string, Object>[]}
//   static List<Map<String, (String, ModelClassInfo)>>? get MODEL_CLASS_MAPPINGS;
//
//   /// Whether to attempt to instantiate the base class (`PretrainedModel`) if
//   /// the model type is not found in the mapping.
//   static bool BASE_IF_FAIL = false;
//
//   static String? name;
// }

/// Dart does not support extending static methods from a class. Therefore, this
/// is handled with the help of some wrapper classes and runtime checks. This
/// class is used as a way to access a particular classes methods, properties,
/// etc. as needed.
class PretrainedReflection {
  final Type name;

  /// Mapping from model type to model class.
  final List<Map<String, (String, ModelClassInfo)>>? MODEL_CLASS_MAPPINGS;

  /// Whether to attempt to instantiate the base class (`PretrainedModel`) if
  /// the model type is not found in the mapping.
  final bool BASE_IF_FAIL;

  PretrainedReflection({
    required this.name,
    this.MODEL_CLASS_MAPPINGS,
    this.BASE_IF_FAIL = false,
  });
}

/// Base class of all AutoModels. Contains the `from_pretrained` function
/// which is used to instantiate pretrained models.
///
/// Dart does not support extending static methods from a class. Therefore, this
/// is handled with the help of some wrapper classes and runtime checks.
class PretrainedMixin {
  /// @type {typeof PreTrainedModel.from_pretrained}
  static Future<PreTrainedModel> from_pretrained(
    PretrainedReflection reflection,
    String pretrained_model_name_or_path,
    [PretrainedOptions? options]
  ) async {
    options = PretrainedModelOptions.fromJson(options?.toJson() ?? {});
    options.config = await AutoConfig.from_pretrained(pretrained_model_name_or_path, options);

    if (reflection.MODEL_CLASS_MAPPINGS == null) {
      throw UnimplementedError("`MODEL_CLASS_MAPPINGS` not implemented for this type of `AutoClass`: ${reflection.name}");
    }

    final model_type = options.config?.model_type;
    for (final MODEL_CLASS_MAPPING in reflection.MODEL_CLASS_MAPPINGS!) {
      (String, ModelClassInfo)? modelInfo = MODEL_CLASS_MAPPING[model_type];

      if (modelInfo == null) {
        // As a fallback, we check if model_type is specified as the exact class
        for (final cls in MODEL_CLASS_MAPPING.values) {
          if (cls.$1 == model_type) {
            modelInfo = cls;
            break;
          }
        }

        if (modelInfo == null) continue; // Item not found in this mapping
      }

      return await modelInfo.$2.from_pretrained(
        pretrained_model_name_or_path,
        options as PretrainedModelOptions,
      );
    }

    if (reflection.BASE_IF_FAIL) {
      if (!(CUSTOM_ARCHITECTURES.containsKey(model_type))) {
        // console.warn
        print('Unknown model class "$model_type", attempting to construct from base class.');
      }

      return await PreTrainedModel.default_from_pretrained(
        pretrained_model_name_or_path,
        options as PretrainedModelOptions,
      );
    } else {
      throw Exception('Unsupported model type: $model_type');
    }
  }
}

class ModelClassInfo {
  final Type cls;

  final Future<PreTrainedModel> Function(
    String pretrained_model_name_or_path,
    PretrainedModelOptions? options,
  ) from_pretrained;

  const ModelClassInfo(this.cls, this.from_pretrained);
}

final Map<String, (String, ModelClassInfo)> MODEL_MAPPING_NAMES_ENCODER_ONLY = {
  'bert': ('BertModel', BertModel.modelClassInfo),
  // 'modernbert': ('ModernBertModel', ModernBertModel.modelClassInfo),
  // 'nomic_bert': ('NomicBertModel', NomicBertModel.modelClassInfo),
  // 'roformer': ('RoFormerModel', RoFormerModel.modelClassInfo),
  // 'electra': ('ElectraModel', ElectraModel.modelClassInfo),
  // 'esm': ('EsmModel', EsmModel.modelClassInfo),
  // 'convbert': ('ConvBertModel', ConvBertModel.modelClassInfo),
  // 'camembert': ('CamembertModel', CamembertModel.modelClassInfo),
  // 'deberta': ('DebertaModel', DebertaModel.modelClassInfo),
  // 'deberta-v2': ('DebertaV2Model', DebertaV2Model.modelClassInfo),
  // 'mpnet': ('MPNetModel', MPNetModel.modelClassInfo),
  // 'albert': ('AlbertModel', AlbertModel.modelClassInfo),
  // 'distilbert': ('DistilBertModel', DistilBertModel.modelClassInfo),
  // 'roberta': ('RobertaModel', RobertaModel.modelClassInfo),
  // 'xlm': ('XLMModel', XLMModel.modelClassInfo),
  'xlm-roberta': ('XLMRobertaModel', XLMRobertaModel.modelClassInfo),
  // 'clap': ('ClapModel', ClapModel.modelClassInfo),
  // 'clip': ('CLIPModel', CLIPModel.modelClassInfo),
  // 'clipseg': ('CLIPSegModel', CLIPSegModel.modelClassInfo),
  // 'chinese_clip': ('ChineseCLIPModel', ChineseCLIPModel.modelClassInfo),
  // 'siglip': ('SiglipModel', SiglipModel.modelClassInfo),
  // 'jina_clip': ('JinaCLIPModel', JinaCLIPModel.modelClassInfo),
  // 'mobilebert': ('MobileBertModel', MobileBertModel.modelClassInfo),
  // 'squeezebert': ('SqueezeBertModel', SqueezeBertModel.modelClassInfo),
  // 'wav2vec2': ('Wav2Vec2Model', Wav2Vec2Model.modelClassInfo),
  // 'wav2vec2-bert': ('Wav2Vec2BertModel', Wav2Vec2BertModel.modelClassInfo),
  // 'unispeech': ('UniSpeechModel', UniSpeechModel.modelClassInfo),
  // 'unispeech-sat': ('UniSpeechSatModel', UniSpeechSatModel.modelClassInfo),
  // 'hubert': ('HubertModel', HubertModel.modelClassInfo),
  // 'wavlm': ('WavLMModel', WavLMModel.modelClassInfo),
  // 'audio-spectrogram-transformer': ('ASTModel', ASTModel.modelClassInfo),
  // 'vits': ('VitsModel', VitsModel.modelClassInfo),
  // 'pyannote': ('PyAnnoteModel', PyAnnoteModel.modelClassInfo),
  // 'wespeaker-resnet': ('WeSpeakerResNetModel', WeSpeakerResNetModel.modelClassInfo),

  // 'detr': ('DetrModel', DetrModel.modelClassInfo),
  // 'rt_detr': ('RTDetrModel', RTDetrModel.modelClassInfo),
  // 'rt_detr_v2': ('RTDetrV2Model', RTDetrV2Model.modelClassInfo),
  // 'rf_detr': ('RFDetrModel', RFDetrModel.modelClassInfo),
  // 'd_fine': ('DFineModel', DFineModel.modelClassInfo),
  // 'table-transformer': ('TableTransformerModel', TableTransformerModel.modelClassInfo),
  // 'vit': ('ViTModel', ViTModel.modelClassInfo),
  // 'ijepa': ('IJepaModel', IJepaModel.modelClassInfo),
  // 'pvt': ('PvtModel', PvtModel.modelClassInfo),
  // 'vit_msn': ('ViTMSNModel', ViTMSNModel.modelClassInfo),
  // 'vit_mae': ('ViTMAEModel', ViTMAEModel.modelClassInfo),
  // 'groupvit': ('GroupViTModel', GroupViTModel.modelClassInfo),
  // 'fastvit': ('FastViTModel', FastViTModel.modelClassInfo),
  // 'mobilevit': ('MobileViTModel', MobileViTModel.modelClassInfo),
  // 'mobilevitv2': ('MobileViTV2Model', MobileViTV2Model.modelClassInfo),
  // 'owlvit': ('OwlViTModel', OwlViTModel.modelClassInfo),
  // 'owlv2': ('Owlv2Model', Owlv2Model.modelClassInfo),
  // 'beit': ('BeitModel', BeitModel.modelClassInfo),
  // 'deit': ('DeiTModel', DeiTModel.modelClassInfo),
  // 'hiera': ('HieraModel', HieraModel.modelClassInfo),
  // 'convnext': ('ConvNextModel', ConvNextModel.modelClassInfo),
  // 'convnextv2': ('ConvNextV2Model', ConvNextV2Model.modelClassInfo),
  // 'dinov2': ('Dinov2Model', Dinov2Model.modelClassInfo),
  // 'dinov2_with_registers': ('Dinov2WithRegistersModel', Dinov2WithRegistersModel.modelClassInfo),
  // 'resnet': ('ResNetModel', ResNetModel.modelClassInfo),
  // 'swin': ('SwinModel', SwinModel.modelClassInfo),
  // 'swin2sr': ('Swin2SRModel', Swin2SRModel.modelClassInfo),
  // 'donut-swin': ('DonutSwinModel', DonutSwinModel.modelClassInfo),
  // 'yolos': ('YolosModel', YolosModel.modelClassInfo),
  // 'dpt': ('DPTModel', DPTModel.modelClassInfo),
  // 'glpn': ('GLPNModel', GLPNModel.modelClassInfo),

  // 'hifigan': ('SpeechT5HifiGan', SpeechT5HifiGan.modelClassInfo),
  // 'efficientnet': ('EfficientNetModel', EfficientNetModel.modelClassInfo),

  // 'decision_transformer': ('DecisionTransformerModel', DecisionTransformerModel.modelClassInfo),
  // 'patchtst': ('PatchTSTForPrediction', PatchTSTModel.modelClassInfo),
  // 'patchtsmixer': ('PatchTSMixerForPrediction', PatchTSMixerModel.modelClassInfo),

  // 'mobilenet_v1': ('MobileNetV1Model', MobileNetV1Model.modelClassInfo),
  // 'mobilenet_v2': ('MobileNetV2Model', MobileNetV2Model.modelClassInfo),
  // 'mobilenet_v3': ('MobileNetV3Model', MobileNetV3Model.modelClassInfo),
  // 'mobilenet_v4': ('MobileNetV4Model', MobileNetV4Model.modelClassInfo),

  // 'maskformer': ('MaskFormerModel', MaskFormerModel.modelClassInfo),
  // 'mgp-str': ('MgpstrForSceneTextRecognition', MgpstrForSceneTextRecognition.modelClassInfo),

  // 'style_text_to_speech_2': ('StyleTextToSpeech2Model', StyleTextToSpeech2Model.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_MAPPING_NAMES_ENCODER_DECODER = {
  // 't5': ('T5Model', T5Model.modelClassInfo),
  // 'longt5': ('LongT5Model', LongT5Model.modelClassInfo),
  // 'mt5': ('MT5Model', MT5Model.modelClassInfo),
  // 'bart': ('BartModel', BartModel.modelClassInfo),
  // 'mbart': ('MBartModel', MBartModel.modelClassInfo),
  // 'marian': ('MarianModel', MarianModel.modelClassInfo),
  // 'whisper': ('WhisperModel', WhisperModel.modelClassInfo),
  // 'm2m_100': ('M2M100Model', M2M100Model.modelClassInfo),
  // 'blenderbot': ('BlenderbotModel', BlenderbotModel.modelClassInfo),
  // 'blenderbot-small': ('BlenderbotSmallModel', BlenderbotSmallModel.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_MAPPING_NAMES_AUTO_ENCODER = {
  // 'mimi': ('MimiModel', MimiModel.modelClassInfo),
  // 'dac': ('DacModel', DacModel.modelClassInfo),
  // 'snac': ('SnacModel', SnacModel.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_MAPPING_NAMES_DECODER_ONLY = {
  // 'bloom': ('BloomModel', BloomModel.modelClassInfo),
  // 'jais': ('JAISModel', JAISModel.modelClassInfo),
  // 'gpt2': ('GPT2Model', GPT2Model.modelClassInfo),
  // 'gptj': ('GPTJModel', GPTJModel.modelClassInfo),
  // 'gpt_bigcode': ('GPTBigCodeModel', GPTBigCodeModel.modelClassInfo),
  // 'gpt_neo': ('GPTNeoModel', GPTNeoModel.modelClassInfo),
  // 'gpt_neox': ('GPTNeoXModel', GPTNeoXModel.modelClassInfo),
  // 'codegen': ('CodeGenModel', CodeGenModel.modelClassInfo),
  // 'llama': ('LlamaModel', LlamaModel.modelClassInfo),
  // 'exaone': ('ExaoneModel', ExaoneModel.modelClassInfo),
  // 'olmo': ('OlmoModel', OlmoModel.modelClassInfo),
  // 'olmo2': ('Olmo2Model', Olmo2Model.modelClassInfo),
  // 'mobilellm': ('MobileLLMModel', MobileLLMModel.modelClassInfo),
  // 'granite': ('GraniteModel', GraniteModel.modelClassInfo),
  'granitemoehybrid': ('GraniteMoeHybridModel', GraniteMoeHybridModel.modelClassInfo),
  // 'cohere': ('CohereModel', CohereModel.modelClassInfo),
  // 'gemma': ('GemmaModel', GemmaModel.modelClassInfo),
  // 'gemma2': ('Gemma2Model', Gemma2Model.modelClassInfo),
  // 'gemma3_text': ('Gemma3Model', Gemma3Model.modelClassInfo),
  // 'helium': ('HeliumModel', HeliumModel.modelClassInfo),
  // 'glm': ('GlmModel', GlmModel.modelClassInfo),
  // 'openelm': ('OpenELMModel', OpenELMModel.modelClassInfo),
  // 'qwen2': ('Qwen2Model', Qwen2Model.modelClassInfo),
  'qwen3': ('Qwen3Model', Qwen3Model.modelClassInfo),
  // 'phi': ('PhiModel', PhiModel.modelClassInfo),
  // 'phi3': ('Phi3Model', Phi3Model.modelClassInfo),
  // 'mpt': ('MptModel', MptModel.modelClassInfo),
  // 'opt': ('OPTModel', OPTModel.modelClassInfo),
  // 'mistral': ('MistralModel', MistralModel.modelClassInfo),
  // 'starcoder2': ('Starcoder2Model', Starcoder2Model.modelClassInfo),
  // 'falcon': ('FalconModel', FalconModel.modelClassInfo),
  // 'stablelm': ('StableLmModel', StableLmModel.modelClassInfo),
};

const Map<String, (String, ModelClassInfo)> MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = {
  // 'speecht5': ('SpeechT5ForSpeechToText', SpeechT5ForSpeechToText.modelClassInfo),
  // 'whisper': ('WhisperForConditionalGeneration', WhisperForConditionalGeneration.modelClassInfo),
  // 'lite-whisper': ('LiteWhisperForConditionalGeneration', LiteWhisperForConditionalGeneration.modelClassInfo),
  // 'moonshine': ('MoonshineForConditionalGeneration', MoonshineForConditionalGeneration.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES = {
  // 'speecht5': ('SpeechT5ForTextToSpeech', SpeechT5ForTextToSpeech.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES = {
  // 'vits': ('VitsModel', VitsModel.modelClassInfo),
  // 'musicgen': ('MusicgenForConditionalGeneration', MusicgenForConditionalGeneration.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = {
  // 'bert': ('BertForSequenceClassification', BertForSequenceClassification.modelClassInfo),
  // 'modernbert': ('ModernBertForSequenceClassification', ModernBertForSequenceClassification.modelClassInfo),
  // 'roformer': ('RoFormerForSequenceClassification', RoFormerForSequenceClassification.modelClassInfo),
  // 'electra': ('ElectraForSequenceClassification', ElectraForSequenceClassification.modelClassInfo),
  // 'esm': ('EsmForSequenceClassification', EsmForSequenceClassification.modelClassInfo),
  // 'convbert': ('ConvBertForSequenceClassification', ConvBertForSequenceClassification.modelClassInfo),
  // 'camembert': ('CamembertForSequenceClassification', CamembertForSequenceClassification.modelClassInfo),
  // 'deberta': ('DebertaForSequenceClassification', DebertaForSequenceClassification.modelClassInfo),
  // 'deberta-v2': ('DebertaV2ForSequenceClassification', DebertaV2ForSequenceClassification.modelClassInfo),
  // 'mpnet': ('MPNetForSequenceClassification', MPNetForSequenceClassification.modelClassInfo),
  // 'albert': ('AlbertForSequenceClassification', AlbertForSequenceClassification.modelClassInfo),
  // 'distilbert': ('DistilBertForSequenceClassification', DistilBertForSequenceClassification.modelClassInfo),
  // 'roberta': ('RobertaForSequenceClassification', RobertaForSequenceClassification.modelClassInfo),
  // 'xlm': ('XLMForSequenceClassification', XLMForSequenceClassification.modelClassInfo),
  // 'xlm-roberta': ('XLMRobertaForSequenceClassification', XLMRobertaForSequenceClassification.modelClassInfo),
  // 'bart': ('BartForSequenceClassification', BartForSequenceClassification.modelClassInfo),
  // 'mbart': ('MBartForSequenceClassification', MBartForSequenceClassification.modelClassInfo),
  // 'mobilebert': ('MobileBertForSequenceClassification', MobileBertForSequenceClassification.modelClassInfo),
  // 'squeezebert': ('SqueezeBertForSequenceClassification', SqueezeBertForSequenceClassification.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = {
  // 'bert': ('BertForTokenClassification', BertForTokenClassification.modelClassInfo),
  // 'modernbert': ('ModernBertForTokenClassification', ModernBertForTokenClassification.modelClassInfo),
  // 'roformer': ('RoFormerForTokenClassification', RoFormerForTokenClassification.modelClassInfo),
  // 'electra': ('ElectraForTokenClassification', ElectraForTokenClassification.modelClassInfo),
  // 'esm': ('EsmForTokenClassification', EsmForTokenClassification.modelClassInfo),
  // 'convbert': ('ConvBertForTokenClassification', ConvBertForTokenClassification.modelClassInfo),
  // 'camembert': ('CamembertForTokenClassification', CamembertForTokenClassification.modelClassInfo),
  // 'deberta': ('DebertaForTokenClassification', DebertaForTokenClassification.modelClassInfo),
  // 'deberta-v2': ('DebertaV2ForTokenClassification', DebertaV2ForTokenClassification.modelClassInfo),
  // 'mpnet': ('MPNetForTokenClassification', MPNetForTokenClassification.modelClassInfo),
  // 'distilbert': ('DistilBertForTokenClassification', DistilBertForTokenClassification.modelClassInfo),
  // 'roberta': ('RobertaForTokenClassification', RobertaForTokenClassification.modelClassInfo),
  // 'xlm': ('XLMForTokenClassification', XLMForTokenClassification.modelClassInfo),
  // 'xlm-roberta': ('XLMRobertaForTokenClassification', XLMRobertaForTokenClassification.modelClassInfo),
};

const Map<String, (String, ModelClassInfo)> MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = {
  // 't5': ('T5ForConditionalGeneration', T5ForConditionalGeneration.modelClassInfo),
  // 'longt5': ('LongT5ForConditionalGeneration', LongT5ForConditionalGeneration.modelClassInfo),
  // 'mt5': ('MT5ForConditionalGeneration', MT5ForConditionalGeneration.modelClassInfo),
  // 'bart': ('BartForConditionalGeneration', BartForConditionalGeneration.modelClassInfo),
  // 'mbart': ('MBartForConditionalGeneration', MBartForConditionalGeneration.modelClassInfo),
  // 'marian': ('MarianMTModel', MarianMTModel.modelClassInfo),
  // 'm2m_100': ('M2M100ForConditionalGeneration', M2M100ForConditionalGeneration.modelClassInfo),
  // 'blenderbot': ('BlenderbotForConditionalGeneration', BlenderbotForConditionalGeneration.modelClassInfo),
  // 'blenderbot-small': ('BlenderbotSmallForConditionalGeneration', BlenderbotSmallForConditionalGeneration.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = {
  // 'bloom': ('BloomForCausalLM', BloomForCausalLM.modelClassInfo),
  'gpt2': ('GPT2LMHeadModel', GPT2LMHeadModel.modelClassInfo),
  // 'jais': ('JAISLMHeadModel', JAISLMHeadModel.modelClassInfo),
  // 'gptj': ('GPTJForCausalLM', GPTJForCausalLM.modelClassInfo),
  // 'gpt_bigcode': ('GPTBigCodeForCausalLM', GPTBigCodeForCausalLM.modelClassInfo),
  // 'gpt_neo': ('GPTNeoForCausalLM', GPTNeoForCausalLM.modelClassInfo),
  // 'gpt_neox': ('GPTNeoXForCausalLM', GPTNeoXForCausalLM.modelClassInfo),
  // 'codegen': ('CodeGenForCausalLM', CodeGenForCausalLM.modelClassInfo),
  // 'llama': ('LlamaForCausalLM', LlamaForCausalLM.modelClassInfo),
  // 'exaone': ('ExaoneForCausalLM', ExaoneForCausalLM.modelClassInfo),
  // 'olmo': ('OlmoForCausalLM', OlmoForCausalLM.modelClassInfo),
  // 'olmo2': ('Olmo2ForCausalLM', Olmo2ForCausalLM.modelClassInfo),
  // 'mobilellm': ('MobileLLMForCausalLM', MobileLLMForCausalLM.modelClassInfo),
  // 'granite': ('GraniteForCausalLM', GraniteForCausalLM.modelClassInfo),
  'granitemoehybrid': ('GraniteMoeHybridForCausalLM', GraniteMoeHybridForCausalLM.modelClassInfo),
  // 'cohere': ('CohereForCausalLM', CohereForCausalLM.modelClassInfo),
  // 'gemma': ('GemmaForCausalLM', GemmaForCausalLM.modelClassInfo),
  // 'gemma2': ('Gemma2ForCausalLM', Gemma2ForCausalLM.modelClassInfo),
  // 'gemma3_text': ('Gemma3ForCausalLM', Gemma3ForCausalLM.modelClassInfo),
  // 'helium': ('HeliumForCausalLM', HeliumForCausalLM.modelClassInfo),
  // 'glm': ('GlmForCausalLM', GlmForCausalLM.modelClassInfo),
  // 'openelm': ('OpenELMForCausalLM', OpenELMForCausalLM.modelClassInfo),
  // 'qwen2': ('Qwen2ForCausalLM', Qwen2ForCausalLM.modelClassInfo),
  'qwen3': ('Qwen3ForCausalLM', Qwen3ForCausalLM.modelClassInfo),
  // 'phi': ('PhiForCausalLM', PhiForCausalLM.modelClassInfo),
  // 'phi3': ('Phi3ForCausalLM', Phi3ForCausalLM.modelClassInfo),
  // 'mpt': ('MptForCausalLM', MptForCausalLM.modelClassInfo),
  // 'opt': ('OPTForCausalLM', OPTForCausalLM.modelClassInfo),
  // 'mbart': ('MBartForCausalLM', MBartForCausalLM.modelClassInfo),
  // 'mistral': ('MistralForCausalLM', MistralForCausalLM.modelClassInfo),
  // 'starcoder2': ('Starcoder2ForCausalLM', Starcoder2ForCausalLM.modelClassInfo),
  // 'falcon': ('FalconForCausalLM', FalconForCausalLM.modelClassInfo),
  // 'trocr': ('TrOCRForCausalLM', TrOCRForCausalLM.modelClassInfo),
  // 'stablelm': ('StableLmForCausalLM', StableLmForCausalLM.modelClassInfo),

  // Also image-text-to-text
  // 'phi3_v': ('Phi3VForCausalLM', Phi3VForCausalLM.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_MULTIMODALITY_MAPPING_NAMES = {
  // 'multi_modality': ('MultiModalityCausalLM', MultiModalityCausalLM.modelClassInfo),
};


final Map<String, (String, ModelClassInfo)> MODEL_FOR_MASKED_LM_MAPPING_NAMES = {
  'bert': ('BertForMaskedLM', BertForMaskedLM.modelClassInfo),
  // 'modernbert': ('ModernBertForMaskedLM', ModernBertForMaskedLM.modelClassInfo),
  // 'roformer': ('RoFormerForMaskedLM', RoFormerForMaskedLM.modelClassInfo),
  // 'electra': ('ElectraForMaskedLM', ElectraForMaskedLM.modelClassInfo),
  // 'esm': ('EsmForMaskedLM', EsmForMaskedLM.modelClassInfo),
  // 'convbert': ('ConvBertForMaskedLM', ConvBertForMaskedLM.modelClassInfo),
  // 'camembert': ('CamembertForMaskedLM', CamembertForMaskedLM.modelClassInfo),
  // 'deberta': ('DebertaForMaskedLM', DebertaForMaskedLM.modelClassInfo),
  // 'deberta-v2': ('DebertaV2ForMaskedLM', DebertaV2ForMaskedLM.modelClassInfo),
  // 'mpnet': ('MPNetForMaskedLM', MPNetForMaskedLM.modelClassInfo),
  // 'albert': ('AlbertForMaskedLM', AlbertForMaskedLM.modelClassInfo),
  // 'distilbert': ('DistilBertForMaskedLM', DistilBertForMaskedLM.modelClassInfo),
  // 'roberta': ('RobertaForMaskedLM', RobertaForMaskedLM.modelClassInfo),
  // 'xlm': ('XLMWithLMHeadModel', XLMWithLMHeadModel.modelClassInfo),
  'xlm-roberta': ('XLMRobertaForMaskedLM', XLMRobertaForMaskedLM.modelClassInfo),
  // 'mobilebert': ('MobileBertForMaskedLM', MobileBertForMaskedLM.modelClassInfo),
  // 'squeezebert': ('SqueezeBertForMaskedLM', SqueezeBertForMaskedLM.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = {
  // 'bert': ('BertForQuestionAnswering', BertForQuestionAnswering.modelClassInfo),
  // 'roformer': ('RoFormerForQuestionAnswering', RoFormerForQuestionAnswering.modelClassInfo),
  // 'electra': ('ElectraForQuestionAnswering', ElectraForQuestionAnswering.modelClassInfo),
  // 'convbert': ('ConvBertForQuestionAnswering', ConvBertForQuestionAnswering.modelClassInfo),
  // 'camembert': ('CamembertForQuestionAnswering', CamembertForQuestionAnswering.modelClassInfo),
  // 'deberta': ('DebertaForQuestionAnswering', DebertaForQuestionAnswering.modelClassInfo),
  // 'deberta-v2': ('DebertaV2ForQuestionAnswering', DebertaV2ForQuestionAnswering.modelClassInfo),
  // 'mpnet': ('MPNetForQuestionAnswering', MPNetForQuestionAnswering.modelClassInfo),
  // 'albert': ('AlbertForQuestionAnswering', AlbertForQuestionAnswering.modelClassInfo),
  // 'distilbert': ('DistilBertForQuestionAnswering', DistilBertForQuestionAnswering.modelClassInfo),
  // 'roberta': ('RobertaForQuestionAnswering', RobertaForQuestionAnswering.modelClassInfo),
  // 'xlm': ('XLMForQuestionAnswering', XLMForQuestionAnswering.modelClassInfo),
  // 'xlm-roberta': ('XLMRobertaForQuestionAnswering', XLMRobertaForQuestionAnswering.modelClassInfo),
  // 'mobilebert': ('MobileBertForQuestionAnswering', MobileBertForQuestionAnswering.modelClassInfo),
  // 'squeezebert': ('SqueezeBertForQuestionAnswering', SqueezeBertForQuestionAnswering.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = {
  // TODO:
  // 'vision-encoder-decoder': ('VisionEncoderDecoderModel', VisionEncoderDecoderModel),
  'idefics3': ('Idefics3ForConditionalGeneration', Idefics3ForConditionalGeneration.modelClassInfo),
  // 'smolvlm': ('SmolVLMForConditionalGeneration', SmolVLMForConditionalGeneration),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = {
  // 'llava': ('LlavaForConditionalGeneration', LlavaForConditionalGeneration.modelClassInfo),
  // 'llava_onevision': ('LlavaOnevisionForConditionalGeneration', LlavaOnevisionForConditionalGeneration.modelClassInfo),
  // 'moondream1': ('Moondream1ForConditionalGeneration', Moondream1ForConditionalGeneration.modelClassInfo),
  // 'florence2': ('Florence2ForConditionalGeneration', Florence2ForConditionalGeneration.modelClassInfo),
  // 'qwen2-vl': ('Qwen2VLForConditionalGeneration', Qwen2VLForConditionalGeneration.modelClassInfo),
  'idefics3': ('Idefics3ForConditionalGeneration', Idefics3ForConditionalGeneration.modelClassInfo),
  // 'smolvlm': ('SmolVLMForConditionalGeneration', SmolVLMForConditionalGeneration.modelClassInfo),
  // 'paligemma': ('PaliGemmaForConditionalGeneration', PaliGemmaForConditionalGeneration.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_AUDIO_TEXT_TO_TEXT_MAPPING_NAMES = {
  // 'ultravox': ('UltravoxModel', UltravoxModel.modelClassInfo),
};


final Map<String, (String, ModelClassInfo)> MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = {
  // 'vision-encoder-decoder': ('VisionEncoderDecoderModel', VisionEncoderDecoderModel.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = {
  // 'vit': ('ViTForImageClassification', ViTForImageClassification.modelClassInfo),
  // 'ijepa': ('IJepaForImageClassification', IJepaForImageClassification.modelClassInfo),
  // 'pvt': ('PvtForImageClassification', PvtForImageClassification.modelClassInfo),
  // 'vit_msn': ('ViTMSNForImageClassification', ViTMSNForImageClassification.modelClassInfo),
  // 'fastvit': ('FastViTForImageClassification', FastViTForImageClassification.modelClassInfo),
  // 'mobilevit': ('MobileViTForImageClassification', MobileViTForImageClassification.modelClassInfo),
  // 'mobilevitv2': ('MobileViTV2ForImageClassification', MobileViTV2ForImageClassification.modelClassInfo),
  // 'beit': ('BeitForImageClassification', BeitForImageClassification.modelClassInfo),
  // 'deit': ('DeiTForImageClassification', DeiTForImageClassification.modelClassInfo),
  // 'hiera': ('HieraForImageClassification', HieraForImageClassification.modelClassInfo),
  // 'convnext': ('ConvNextForImageClassification', ConvNextForImageClassification.modelClassInfo),
  // 'convnextv2': ('ConvNextV2ForImageClassification', ConvNextV2ForImageClassification.modelClassInfo),
  // 'dinov2': ('Dinov2ForImageClassification', Dinov2ForImageClassification.modelClassInfo),
  // 'dinov2_with_registers': ('Dinov2WithRegistersForImageClassification', Dinov2WithRegistersForImageClassification.modelClassInfo),
  // 'resnet': ('ResNetForImageClassification', ResNetForImageClassification.modelClassInfo),
  // 'swin': ('SwinForImageClassification', SwinForImageClassification.modelClassInfo),
  // 'segformer': ('SegformerForImageClassification', SegformerForImageClassification.modelClassInfo),
  // 'efficientnet': ('EfficientNetForImageClassification', EfficientNetForImageClassification.modelClassInfo),
  // 'mobilenet_v1': ('MobileNetV1ForImageClassification', MobileNetV1ForImageClassification.modelClassInfo),
  // 'mobilenet_v2': ('MobileNetV2ForImageClassification', MobileNetV2ForImageClassification.modelClassInfo),
  // 'mobilenet_v3': ('MobileNetV3ForImageClassification', MobileNetV3ForImageClassification.modelClassInfo),
  // 'mobilenet_v4': ('MobileNetV4ForImageClassification', MobileNetV4ForImageClassification.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES = {
  // 'detr': ('DetrForObjectDetection', DetrForObjectDetection.modelClassInfo),
  // 'rt_detr': ('RTDetrForObjectDetection', RTDetrForObjectDetection.modelClassInfo),
  // 'rt_detr_v2': ('RTDetrV2ForObjectDetection', RTDetrV2ForObjectDetection.modelClassInfo),
  // 'rf_detr': ('RFDetrForObjectDetection', RFDetrForObjectDetection.modelClassInfo),
  // 'd_fine': ('DFineForObjectDetection', DFineForObjectDetection.modelClassInfo),
  // 'table-transformer': ('TableTransformerForObjectDetection', TableTransformerForObjectDetection.modelClassInfo),
  // 'yolos': ('YolosForObjectDetection', YolosForObjectDetection.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES = {
  // 'owlvit': ('OwlViTForObjectDetection', OwlViTForObjectDetection.modelClassInfo),
  // 'owlv2': ('Owlv2ForObjectDetection', Owlv2ForObjectDetection.modelClassInfo),
  // 'grounding-dino': ('GroundingDinoForObjectDetection', GroundingDinoForObjectDetection.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES = {
  // TODO: Do not add new models here
  // TODO:
  // 'detr': ('DetrForSegmentation', DetrForSegmentation),
  // 'clipseg': ('CLIPSegForImageSegmentation', CLIPSegForImageSegmentation),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES = {
  // 'segformer': ('SegformerForSemanticSegmentation', SegformerForSemanticSegmentation.modelClassInfo),
  // 'sapiens': ('SapiensForSemanticSegmentation', SapiensForSemanticSegmentation.modelClassInfo),

  // 'swin': ('SwinForSemanticSegmentation', SwinForSemanticSegmentation.modelClassInfo),
  // 'mobilenet_v1': ('MobileNetV1ForSemanticSegmentation', MobileNetV1ForSemanticSegmentation.modelClassInfo),
  // 'mobilenet_v2': ('MobileNetV2ForSemanticSegmentation', MobileNetV2ForSemanticSegmentation.modelClassInfo),
  // 'mobilenet_v3': ('MobileNetV3ForSemanticSegmentation', MobileNetV3ForSemanticSegmentation.modelClassInfo),
  // 'mobilenet_v4': ('MobileNetV4ForSemanticSegmentation', MobileNetV4ForSemanticSegmentation.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES = {
  // 'detr': ('DetrForSegmentation', DetrForSegmentation.modelClassInfo),
  // 'maskformer': ('MaskFormerForInstanceSegmentation', MaskFormerForInstanceSegmentation.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_MASK_GENERATION_MAPPING_NAMES = {
  // 'sam': ('SamModel', SamModel.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_CTC_MAPPING_NAMES = {
  // 'wav2vec2': ('Wav2Vec2ForCTC', Wav2Vec2ForCTC.modelClassInfo),
  // 'wav2vec2-bert': ('Wav2Vec2BertForCTC', Wav2Vec2BertForCTC.modelClassInfo),
  // 'unispeech': ('UniSpeechForCTC', UniSpeechForCTC.modelClassInfo),
  // 'unispeech-sat': ('UniSpeechSatForCTC', UniSpeechSatForCTC.modelClassInfo),
  // 'wavlm': ('WavLMForCTC', WavLMForCTC.modelClassInfo),
  // 'hubert': ('HubertForCTC', HubertForCTC.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = {
  // 'wav2vec2': ('Wav2Vec2ForSequenceClassification', Wav2Vec2ForSequenceClassification.modelClassInfo),
  // 'wav2vec2-bert': ('Wav2Vec2BertForSequenceClassification', Wav2Vec2BertForSequenceClassification.modelClassInfo),
  // 'unispeech': ('UniSpeechForSequenceClassification', UniSpeechForSequenceClassification.modelClassInfo),
  // 'unispeech-sat': ('UniSpeechSatForSequenceClassification', UniSpeechSatForSequenceClassification.modelClassInfo),
  // 'wavlm': ('WavLMForSequenceClassification', WavLMForSequenceClassification.modelClassInfo),
  // 'hubert': ('HubertForSequenceClassification', HubertForSequenceClassification.modelClassInfo),
  // 'audio-spectrogram-transformer': ('ASTForAudioClassification', ASTForAudioClassification.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES = {
  // 'wavlm': ('WavLMForXVector', WavLMForXVector.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES = {
  // 'unispeech-sat': ('UniSpeechSatForAudioFrameClassification', UniSpeechSatForAudioFrameClassification.modelClassInfo),
  // 'wavlm': ('WavLMForAudioFrameClassification', WavLMForAudioFrameClassification.modelClassInfo),
  // 'wav2vec2': ('Wav2Vec2ForAudioFrameClassification', Wav2Vec2ForAudioFrameClassification.modelClassInfo),
  // 'pyannote': ('PyAnnoteForAudioFrameClassification', PyAnnoteForAudioFrameClassification.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_IMAGE_MATTING_MAPPING_NAMES = {
  // 'vitmatte': ('VitMatteForImageMatting', VitMatteForImageMatting.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_TIME_SERIES_PREDICTION_MAPPING_NAMES = {
  // 'patchtst': ('PatchTSTForPrediction', PatchTSTForPrediction.modelClassInfo),
  // 'patchtsmixer': ('PatchTSMixerForPrediction', PatchTSMixerForPrediction.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES = {
  // 'swin2sr': ('Swin2SRForImageSuperResolution', Swin2SRForImageSuperResolution.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES = {
  // 'dpt': ('DPTForDepthEstimation', DPTForDepthEstimation.modelClassInfo),
  // 'depth_anything': ('DepthAnythingForDepthEstimation', DepthAnythingForDepthEstimation.modelClassInfo),
  // 'glpn': ('GLPNForDepthEstimation', GLPNForDepthEstimation.modelClassInfo),
  // 'sapiens': ('SapiensForDepthEstimation', SapiensForDepthEstimation.modelClassInfo),
  // 'depth_pro': ('DepthProForDepthEstimation', DepthProForDepthEstimation.modelClassInfo),
  // 'metric3d': ('Metric3DForDepthEstimation', Metric3DForDepthEstimation.modelClassInfo),
  // 'metric3dv2': ('Metric3Dv2ForDepthEstimation', Metric3Dv2ForDepthEstimation.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_NORMAL_ESTIMATION_MAPPING_NAMES = {
  // 'sapiens': ('SapiensForNormalEstimation', SapiensForNormalEstimation.modelClassInfo),
};

final Map<String, (String, ModelClassInfo)> MODEL_FOR_POSE_ESTIMATION_MAPPING_NAMES = {
  // 'vitpose': ('VitPoseForPoseEstimation', VitPoseForPoseEstimation.modelClassInfo),
};

// NOTE: This is custom to Transformers.js, and is necessary because certain models
// (e.g., CLIP) are split into vision and text components
final Map<String, (String, ModelClassInfo)> MODEL_FOR_IMAGE_FEATURE_EXTRACTION_MAPPING_NAMES = {
  // 'clip': ('CLIPVisionModelWithProjection', CLIPVisionModelWithProjection.modelClassInfo),
  // 'siglip': ('SiglipVisionModel', SiglipVisionModel.modelClassInfo),
  // 'jina_clip': ('JinaCLIPVisionModel', JinaCLIPVisionModel.modelClassInfo),
};

final List<(Map<String, (String, ModelClassInfo)>, MODEL_TYPES)> MODEL_CLASS_TYPE_MAPPING = [
  // MODEL_MAPPING_NAMES:
  (MODEL_MAPPING_NAMES_ENCODER_ONLY, MODEL_TYPES.EncoderOnly),
  (MODEL_MAPPING_NAMES_ENCODER_DECODER, MODEL_TYPES.EncoderDecoder),
  (MODEL_MAPPING_NAMES_DECODER_ONLY, MODEL_TYPES.DecoderOnly),
  (MODEL_MAPPING_NAMES_AUTO_ENCODER, MODEL_TYPES.AutoEncoder),

  (MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES, MODEL_TYPES.Seq2Seq),
  (MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES, MODEL_TYPES.Seq2Seq),
  (MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_TYPES.DecoderOnly),
  (MODEL_FOR_MULTIMODALITY_MAPPING_NAMES, MODEL_TYPES.MultiModality),
  (MODEL_FOR_MASKED_LM_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES, MODEL_TYPES.Vision2Seq),
  (MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES, MODEL_TYPES.ImageTextToText),
  (MODEL_FOR_AUDIO_TEXT_TO_TEXT_MAPPING_NAMES, MODEL_TYPES.AudioTextToText),
  (MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_IMAGE_MATTING_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_TIME_SERIES_PREDICTION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_NORMAL_ESTIMATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_POSE_ESTIMATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_MASK_GENERATION_MAPPING_NAMES, MODEL_TYPES.MaskGeneration),
  (MODEL_FOR_CTC_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES, MODEL_TYPES.Seq2Seq),
  (MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
  (MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),

  // Custom:
  (MODEL_FOR_IMAGE_FEATURE_EXTRACTION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly),
];
void setupModelClassTypeMappings() {
  for (final (mappings, type) in MODEL_CLASS_TYPE_MAPPING) {
    for (final (name, model) in mappings.values) {
      MODEL_TYPE_MAPPING[name] = type;
      MODEL_CLASS_TO_NAME_MAPPING[model.cls] = name;
      MODEL_NAME_TO_CLASS_MAPPING[name] = model.cls;
    }
  }
}

const List<(String, Type, MODEL_TYPES)> CUSTOM_MAPPING = [
  // OVERRIDE:
  // TODO: Refactor to allow class to specify model
  // ('MusicgenForConditionalGeneration', MusicgenForConditionalGeneration, MODEL_TYPES.Musicgen),
  // ('Phi3VForCausalLM', Phi3VForCausalLM, MODEL_TYPES.Phi3V),

  // ('CLIPTextModelWithProjection', CLIPTextModelWithProjection, MODEL_TYPES.EncoderOnly),
  // ('SiglipTextModel', SiglipTextModel, MODEL_TYPES.EncoderOnly),
  // ('JinaCLIPTextModel', JinaCLIPTextModel, MODEL_TYPES.EncoderOnly),
  // ('ClapTextModelWithProjection', ClapTextModelWithProjection, MODEL_TYPES.EncoderOnly),
  // ('ClapAudioModelWithProjection', ClapAudioModelWithProjection, MODEL_TYPES.EncoderOnly),

  // ('DacEncoderModel', DacEncoderModel, MODEL_TYPES.EncoderOnly),
  // ('DacDecoderModel', DacDecoderModel, MODEL_TYPES.EncoderOnly),
  // ('MimiEncoderModel', MimiEncoderModel, MODEL_TYPES.EncoderOnly),
  // ('MimiDecoderModel', MimiDecoderModel, MODEL_TYPES.EncoderOnly),
  // ('SnacEncoderModel', SnacEncoderModel, MODEL_TYPES.EncoderOnly),
  // ('SnacDecoderModel', SnacDecoderModel, MODEL_TYPES.EncoderOnly),
];
void setupCustomMapping() {
  for (final (name, model, type) in CUSTOM_MAPPING) {
    MODEL_TYPE_MAPPING[name] = type;
    MODEL_CLASS_TO_NAME_MAPPING[model] = name;
    MODEL_NAME_TO_CLASS_MAPPING[name] = model;
  }
}

final Map<String, Map<String, (String, ModelClassInfo)>> CUSTOM_ARCHITECTURES = {
  'modnet': MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES,
  'birefnet': MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES,
  'isnet': MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES,
  'ben': MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES,
};
void setupCustomArchitecturesMapping() {
  for (final e in CUSTOM_ARCHITECTURES.entries) {
    final name = e.key;
    final mapping = e.value;

    mapping[name] = ('PreTrainedModel', ModelClassInfo(PreTrainedModel, PreTrainedModel.default_from_pretrained));
    MODEL_TYPE_MAPPING[name] = MODEL_TYPES.EncoderOnly;
    MODEL_CLASS_TO_NAME_MAPPING[PreTrainedModel] = name;
    MODEL_NAME_TO_CLASS_MAPPING[name] = PreTrainedModel;
  }
}

/// Call to make sure the mappings for models are configured. This is done for
/// you by calling [Transformers.ensureInitialized].
void setupModelMappings() {
  setupModelClassTypeMappings();
  setupCustomMapping();
  setupCustomArchitecturesMapping();
}

/// Helper class which is used to instantiate pretrained models with the `from_pretrained` function.
/// The chosen model class is determined by the type specified in the model config.
///
/// @example
/// let model = await AutoModel.from_pretrained('Xenova/bert-base-uncased');
class AutoModel {
  static final List<Map<String, (String, ModelClassInfo)>>? MODEL_CLASS_MAPPINGS = MODEL_CLASS_TYPE_MAPPING
      .map((x) => x.$1).toList();

  static bool BASE_IF_FAIL = true;

  static final _reflection = PretrainedReflection(
    name: AutoModel,
    MODEL_CLASS_MAPPINGS: MODEL_CLASS_MAPPINGS,
    BASE_IF_FAIL: BASE_IF_FAIL,
  );

  static Future<PreTrainedModel> from_pretrained(
    String pretrained_model_name_or_path,
    [PretrainedOptions? options]
  ) async {
    return await PretrainedMixin.from_pretrained(_reflection, pretrained_model_name_or_path, options);
  }
}

/// Helper class which is used to instantiate pretrained causal language models with the `from_pretrained` function.
/// The chosen model class is determined by the type specified in the model config.
///
/// @example
/// let model = await AutoModelForCausalLM.from_pretrained('Xenova/gpt2');
class AutoModelForCausalLM {
  static final List<Map<String, (String, ModelClassInfo)>>? MODEL_CLASS_MAPPINGS = [MODEL_FOR_CAUSAL_LM_MAPPING_NAMES];

  static bool BASE_IF_FAIL = false;

  static final _reflection = PretrainedReflection(
    name: AutoModelForCausalLM,
    MODEL_CLASS_MAPPINGS: MODEL_CLASS_MAPPINGS,
    BASE_IF_FAIL: BASE_IF_FAIL,
  );

  static Future<PreTrainedModel> from_pretrained(
    String pretrained_model_name_or_path,
    [PretrainedOptions? options]
  ) async {
    return await PretrainedMixin.from_pretrained(_reflection, pretrained_model_name_or_path, options);
  }
}

/// Helper class which is used to instantiate pretrained masked language models with the `from_pretrained` function.
/// The chosen model class is determined by the type specified in the model config.
///
/// @example
/// let model = await AutoModelForMaskedLM.from_pretrained('Xenova/bert-base-uncased');
class AutoModelForMaskedLM {
  static final List<Map<String, (String, ModelClassInfo)>>? MODEL_CLASS_MAPPINGS = [MODEL_FOR_MASKED_LM_MAPPING_NAMES];

  static bool BASE_IF_FAIL = false;

  static final _reflection = PretrainedReflection(
    name: AutoModelForMaskedLM,
    MODEL_CLASS_MAPPINGS: MODEL_CLASS_MAPPINGS,
    BASE_IF_FAIL: BASE_IF_FAIL,
  );

  static Future<PreTrainedModel> from_pretrained(
    String pretrained_model_name_or_path,
    [PretrainedOptions? options]
  ) async {
    return await PretrainedMixin.from_pretrained(_reflection, pretrained_model_name_or_path, options);
  }
}

// /// Helper class which is used to instantiate pretrained token classification models with the `from_pretrained` function.
// /// The chosen model class is determined by the type specified in the model config.
// ///
// /// @example
// /// let model = await AutoModelForTokenClassification.from_pretrained('Xenova/distilbert-base-multilingual-cased-ner-hrl');
// class AutoModelForTokenClassification extends PretrainedMixin {
//   static MODEL_CLASS_MAPPINGS = [MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES];
// }

/// Helper class which is used to instantiate pretrained vision-to-sequence models with the `from_pretrained` function.
/// The chosen model class is determined by the type specified in the model config.
///
/// @example
/// let model = await AutoModelForVision2Seq.from_pretrained('Xenova/vit-gpt2-image-captioning');
class AutoModelForVision2Seq {
  static final List<Map<String, (String, ModelClassInfo)>>? MODEL_CLASS_MAPPINGS = [MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES];

  static bool BASE_IF_FAIL = false;

  static final _reflection = PretrainedReflection(
    name: AutoModelForVision2Seq,
    MODEL_CLASS_MAPPINGS: MODEL_CLASS_MAPPINGS,
    BASE_IF_FAIL: BASE_IF_FAIL,
  );

  static Future<PreTrainedModel> from_pretrained(
    String pretrained_model_name_or_path,
    [PretrainedOptions? options]
  ) async {
    return await PretrainedMixin.from_pretrained(_reflection, pretrained_model_name_or_path, options);
  }
}

/// Base class for masked language models outputs.
class MaskedLMOutput extends ModelOutput {
  Tensor logits;

  /// @param {Object} output The output of the model.
  /// @param {Tensor} output.logits Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
  MaskedLMOutput(Map<String, dynamic> output) : logits = output['logits'];
}
