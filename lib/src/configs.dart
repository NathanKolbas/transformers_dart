// ignore_for_file: non_constant_identifier_names

import 'dart:convert';

import 'package:transformers/src/utils/core.dart';
import 'package:transformers/src/utils/devices.dart';
import 'package:transformers/src/utils/dtypes.dart';
import 'package:transformers/src/utils/hub.dart';
import 'package:transformers/src/utils/tensor.dart';


/// Loads a config from the specified path.
/// @param {string} pretrained_model_name_or_path The path to the config directory.
/// @param {PretrainedOptions} options Additional options for loading the config.
/// @returns {Promise<Object>} A promise that resolves with information about the loaded config.
Future<Map<String, dynamic>> loadConfig(String pretrained_model_name_or_path, PretrainedOptions options) async {
  return await getModelJSON(pretrained_model_name_or_path, 'config.json', true, options);
}

///
/// @param {PretrainedConfig} config
/// @returns {Object} The normalized configuration.
Map<String, dynamic> getNormalizedConfig(Map<String, dynamic> config) {
  final mapping = {};

  Map<String, dynamic> init_normalized_config = {};
  switch (config['model_type']) {
  // Sub-configs
    case 'llava':
    case 'paligemma':
    case 'gemma3':
    case 'florence2':
    case 'llava_onevision':
    case 'idefics3':
    case 'ultravox':
    case 'smolvlm':
      init_normalized_config = getNormalizedConfig(config['text_config']);
      break;
    case 'moondream1':
      init_normalized_config = getNormalizedConfig(config['phi_config']);
      break;
    case 'musicgen':
      init_normalized_config = getNormalizedConfig(config['decoder']);
      break;
    case 'multi_modality':
      init_normalized_config = getNormalizedConfig(config['language_config']);
      break;

  // Decoder-only models
    case 'gpt2':
    case 'gptj':
    case 'jais':
    case 'codegen':
    case 'gpt_bigcode':
      mapping['num_heads'] = 'n_head';
      mapping['num_layers'] = 'n_layer';
      mapping['hidden_size'] = 'n_embd';
      break;
    case 'gpt_neox':
    case 'stablelm':
    case 'opt':
    case 'falcon':
      mapping['num_heads'] = 'num_attention_heads';
      mapping['num_layers'] = 'num_hidden_layers';
      mapping['hidden_size'] = 'hidden_size';
      break;
    case 'llama':
    case 'olmo':
    case 'olmo2':
    case 'mobilellm':
    case 'granite':
    case 'granitemoehybrid':
    case 'cohere':
    case 'mistral':
    case 'starcoder2':
    case 'qwen2':
    case 'qwen2_vl':
    case 'phi':
    case 'phi3':
    case 'phi3_v':
      mapping['num_heads'] = 'num_key_value_heads';
      mapping['num_layers'] = 'num_hidden_layers';
      mapping['hidden_size'] = 'hidden_size';
      mapping['num_attention_heads'] = 'num_attention_heads';
      break;
    case 'qwen3':
    case 'gemma':
    case 'gemma2':
    case 'gemma3_text':
    case 'glm':
    case 'helium':
      mapping['num_heads'] = 'num_key_value_heads';
      mapping['num_layers'] = 'num_hidden_layers';
      mapping['dim_kv'] = 'head_dim';
      break;
    case 'openelm':
      mapping['num_heads'] = 'num_kv_heads';
      mapping['num_layers'] = 'num_transformer_layers';
      mapping['dim_kv'] = 'head_dim';
      break;
    case 'gpt_neo':
    case 'donut-swin':
      mapping['num_heads'] = 'num_heads';
      mapping['num_layers'] = 'num_layers';
      mapping['hidden_size'] = 'hidden_size';
      break;
    case 'bloom':
      mapping['num_heads'] = 'n_head';
      mapping['num_layers'] = 'n_layer';
      mapping['hidden_size'] = 'hidden_size';
      break;
    case 'mpt':
      mapping['num_heads'] = 'n_heads';
      mapping['num_layers'] = 'n_layers';
      mapping['hidden_size'] = 'd_model';
      break;
    case 'exaone':
      mapping['num_heads'] = 'num_key_value_heads';
      mapping['num_layers'] = 'num_layers';
      mapping['dim_kv'] = 'head_dim';
      mapping['num_attention_heads'] = 'num_attention_heads';
      break;

  // Encoder-decoder models
    case 't5':
    case 'mt5':
    case 'longt5':
      mapping['num_decoder_layers'] = 'num_decoder_layers';
      mapping['num_decoder_heads'] = 'num_heads';
      mapping['decoder_dim_kv'] = 'd_kv';
      mapping['num_encoder_layers'] = 'num_layers';
      mapping['num_encoder_heads'] = 'num_heads';
      mapping['encoder_dim_kv'] = 'd_kv';
      break;
    case 'bart':
    case 'mbart':
    case 'marian':
    case 'whisper':
    case 'lite-whisper':
    case 'm2m_100':
    case 'blenderbot':
    case 'blenderbot-small':
    case 'florence2_language':
      mapping['num_decoder_layers'] = 'decoder_layers';
      mapping['num_decoder_heads'] = 'decoder_attention_heads';
      mapping['decoder_hidden_size'] = 'd_model';
      mapping['num_encoder_layers'] = 'encoder_layers';
      mapping['num_encoder_heads'] = 'encoder_attention_heads';
      mapping['encoder_hidden_size'] = 'd_model';
      break;
    case 'speecht5':
      mapping['num_decoder_layers'] = 'decoder_layers';
      mapping['num_decoder_heads'] = 'decoder_attention_heads';
      mapping['decoder_hidden_size'] = 'hidden_size';
      mapping['num_encoder_layers'] = 'encoder_layers';
      mapping['num_encoder_heads'] = 'encoder_attention_heads';
      mapping['encoder_hidden_size'] = 'hidden_size';
      break;
    case 'trocr':
      mapping['num_encoder_layers'] = mapping['num_decoder_layers'] = 'decoder_layers';
      mapping['num_encoder_heads'] = mapping['num_decoder_heads'] = 'decoder_attention_heads';
      mapping['encoder_hidden_size'] = mapping['decoder_hidden_size'] = 'd_model';
      break;
    case 'musicgen_decoder':
      mapping['num_encoder_layers'] = mapping['num_decoder_layers'] = 'num_hidden_layers';
      mapping['num_encoder_heads'] = mapping['num_decoder_heads'] = 'num_attention_heads';
      mapping['encoder_hidden_size'] = mapping['decoder_hidden_size'] = 'hidden_size';
      break;
    case 'moonshine':
      mapping['num_decoder_layers'] = 'decoder_num_hidden_layers';
      mapping['num_decoder_heads'] = 'decoder_num_key_value_heads';
      mapping['num_encoder_layers'] = 'encoder_num_hidden_layers';
      mapping['num_encoder_heads'] = 'encoder_num_key_value_heads';
      mapping['encoder_hidden_size'] = mapping['decoder_hidden_size'] = 'hidden_size';
      break;
    case 'vision-encoder-decoder':
      final decoderConfig = getNormalizedConfig(config['decoder']);

      final add_encoder_pkv = decoderConfig.containsKey('num_decoder_layers');
      final result = pick(config, ['model_type', 'is_encoder_decoder']);
      if (add_encoder_pkv) {
        // Decoder is part of an encoder-decoder model
        result['num_decoder_layers'] = decoderConfig['num_decoder_layers'];
        result['num_decoder_heads'] = decoderConfig['num_decoder_heads'];
        result['decoder_hidden_size'] = decoderConfig['decoder_hidden_size'];

        result['num_encoder_layers'] = decoderConfig['num_encoder_layers'];
        result['num_encoder_heads'] = decoderConfig['num_encoder_heads'];
        result['encoder_hidden_size'] = decoderConfig['encoder_hidden_size'];
      } else {
        // Decoder is a decoder-only model
        result['num_layers'] = decoderConfig['num_layers'];
        result['num_heads'] = decoderConfig['num_heads'];
        result['hidden_size'] = decoderConfig['hidden_size'];
      }
      return result;

  }

  // NOTE: If `num_attention_heads` is not set, it is assumed to be equal to `num_heads`
  final normalized_config = {
    ...init_normalized_config,
    ...pick(config, ['model_type', 'multi_query', 'is_encoder_decoder']),
  };
  for (final key in mapping.keys) {
    normalized_config[key] = config[mapping[key]];
  }
  return normalized_config;
}

/// @param {PretrainedConfig} config
/// @returns {Record<string, number[]>}
Map<String, List<int>> getKeyValueShapes(PretrainedConfig? config, {
  String prefix = 'past_key_values',
  int batch_size = 1,
}) {
  /** @type {Record<string, number[]>} */
  final Map<String, List<int>> decoderFeeds = {};
  final Map<String, dynamic> normalized_config = config?.normalized_config ?? {};

  if (normalized_config['is_encoder_decoder'] == true && (
      normalized_config.containsKey('num_encoder_heads') && normalized_config.containsKey('num_decoder_heads')
  )) {
    final int encoder_dim_kv = normalized_config['encoder_dim_kv'] ?? (
        normalized_config['encoder_hidden_size'] ~/ normalized_config['num_encoder_heads']
    );
    final int decoder_dim_kv = normalized_config['decoder_dim_kv'] ?? (
        normalized_config['decoder_hidden_size'] ~/ normalized_config['num_decoder_heads']
    );

    final List<int> encoder_dims = [batch_size, normalized_config['num_encoder_heads'], 0, encoder_dim_kv];
    final List<int> decoder_dims = [batch_size, normalized_config['num_decoder_heads'], 0, decoder_dim_kv];
    for (int i=0; i < normalized_config['num_decoder_layers']; ++i) {
      decoderFeeds['$prefix.$i.encoder.key'] = encoder_dims;
      decoderFeeds['$prefix.$i.encoder.value'] = encoder_dims;
      decoderFeeds['$prefix.$i.decoder.key'] = decoder_dims;
      decoderFeeds['$prefix.$i.decoder.value'] = decoder_dims;
    }
  } else { // Decoders
    final int num_layers = normalized_config['num_layers'];

    if (normalized_config['model_type'] == 'falcon') {
      final int num_heads = normalized_config['num_heads'];
      final int dim_kv = normalized_config['dim_kv'] ?? (
          normalized_config['hidden_size'] ~/ (normalized_config['num_attention_heads'] ?? num_heads)
      );

      // NOTE: Custom implementation for Falcon
      final List<int> dims = [batch_size * num_heads, 0, dim_kv];
      for (int i=0; i < num_layers; ++i) {
        decoderFeeds['$prefix.$i.key'] = dims;
        decoderFeeds['$prefix.$i.value'] = dims;
      }
    } else if (normalized_config['multi_query'] != null && normalized_config['multi_query'] == true) { // e.g., for `gpt_bigcode`
      final int num_heads = normalized_config['num_heads'];
      final int dim_kv = normalized_config['dim_kv'] ?? (
          normalized_config['hidden_size'] ~/ (normalized_config['num_attention_heads'] ?? num_heads)
      );

      final List<int> dims = [batch_size * num_heads, 0, 2 * dim_kv];

      for (int i=0; i < num_layers; ++i) {
        decoderFeeds['$prefix.$i.key_value'] = dims;
      }
    } else if (normalized_config['model_type'] == 'bloom') {
      // NOTE: Custom implementation for Bloom

      final int num_heads = normalized_config['num_heads'];
      final int dim_kv = normalized_config['dim_kv'] ?? (
          normalized_config['hidden_size'] ~/ (normalized_config['num_attention_heads'] ?? num_heads)
      );

      final List<int> keyDims = [batch_size * num_heads, dim_kv, 0]; // [batch_size x num_heads,64,past_sequence_length]
      final List<int> valueDims = [batch_size * num_heads, 0, dim_kv]; // [batch_size x num_heads,past_sequence_length,64]
      for (int i=0; i < num_layers; ++i) {
        decoderFeeds['$prefix.$i.key'] = keyDims;
        decoderFeeds['$prefix.$i.value'] = valueDims;
      }
    } else if (normalized_config['model_type'] == 'openelm') {
      final List<int> num_heads = normalized_config['num_heads'];
      final int dim_kv = normalized_config['dim_kv'] ?? (
          normalized_config['hidden_size'] ~/ normalized_config['num_attention_heads']
      );

      for (int i = 0; i < num_layers; ++i) {
        final List<int> dims = [batch_size, num_heads[i], 0, dim_kv];

        decoderFeeds['$prefix.$i.key'] = dims;
        decoderFeeds['$prefix.$i.value'] = dims;
      }
    } else { // Decoder-only
      final int num_heads = normalized_config['num_heads'];
      final int dim_kv = normalized_config['dim_kv'] ?? (
          normalized_config['hidden_size'] ~/ (normalized_config['num_attention_heads'] ?? num_heads)
      );

      final List<int> dims = [batch_size, num_heads, 0, dim_kv];
      for (int i=0; i < num_layers; ++i) {
        decoderFeeds['$prefix.$i.key'] = dims;
        decoderFeeds['$prefix.$i.value'] = dims;
      }
    }
  }

  return decoderFeeds;
}

/// Base class for all configuration classes. For more information, see the corresponding
/// [Python documentation](https://huggingface.co/docs/transformers/main/en/main_classes/configuration#transformers.PretrainedConfig).
class PretrainedConfig {
  // NOTE: Typo in original

  String? model_type;

  bool is_encoder_decoder;

  int? max_position_embeddings;

  /** @type {TransformersJSConfig} */
  // 'transformers.js_config';
  TransformersJSConfig transformersJsConfig;

  Map<String, dynamic> normalized_config;

  int? image_token_index;
  int? num_image_tokens;

  /// The JSON that was used to create this config. May contain more data than
  /// what is hardcoded in this class.
  Map<String, dynamic> _rawJson;

  /// Create a new PreTrainedTokenizer instance.
  /// @param {Object} configJSON The JSON of the config.
  PretrainedConfig(Map<String, dynamic> configJSON)
      : model_type = configJSON['model_type'],
        is_encoder_decoder = configJSON['is_encoder_decoder'] ?? false,
        max_position_embeddings = configJSON['max_position_embeddings'],
        transformersJsConfig = configJSON['transformers.js_config'] == null
            ? TransformersJSConfig()
            : TransformersJSConfig.fromJson(configJSON['transformers.js_config']),
        normalized_config = getNormalizedConfig(configJSON),
        image_token_index = configJSON['image_token_index'],
        num_image_tokens = configJSON['num_image_tokens'],
        _rawJson = { ...configJSON } {
    _rawJson['is_encoder_decoder'] = is_encoder_decoder;
    _rawJson['normalized_config'] = normalized_config;
  }

  factory PretrainedConfig.fromJson(Map<String, dynamic> json) => PretrainedConfig(json);

  /// Loads a pre-trained config from the given `pretrained_model_name_or_path`.
  ///
  /// @param {string} pretrained_model_name_or_path The path to the pre-trained config.
  /// @param {PretrainedOptions} options Additional options for loading the config.
  /// @throws {Error} Throws an error if the config.json is not found in the `pretrained_model_name_or_path`.
  ///
  /// @returns {Promise<PretrainedConfig>} A new instance of the `PretrainedConfig` class.
  static Future<PretrainedConfig> from_pretrained(String pretrained_model_name_or_path, [PretrainedOptions? options]) async {
    final progress_callback = options?.progress_callback;
    PretrainedConfig? config = options?.config;
    final cache_dir = options?.cache_dir;
    final local_files_only = options?.local_files_only ?? false;
    final revision = options?.revision ?? 'main';

    // if (config != null && config is! PretrainedConfig) {
    //   config = PretrainedConfig(config);
    // }

    final Map<String, dynamic> data = config?.toJson() ?? await loadConfig(pretrained_model_name_or_path, PretrainedOptions(
      progress_callback: progress_callback,
      config: config,
      cache_dir: cache_dir,
      local_files_only: local_files_only,
      revision: revision,
    ));
    return PretrainedConfig.fromJson(data);
  }

  dynamic operator [](String key) => _rawJson[key];

  Map<String, dynamic> toJson() => { ..._rawJson };

  @override
  String toString() => jsonEncode(toJson());
}

/// Helper class which is used to instantiate pretrained configs with the `from_pretrained` function.
///
/// @example
/// const config = await AutoConfig.from_pretrained('Xenova/bert-base-uncased');
class AutoConfig {
  /// @type {typeof PretrainedConfig.from_pretrained}
  static Future<PretrainedConfig> from_pretrained(String pretrained_model_name_or_path, [PretrainedOptions? options]) async {
    return PretrainedConfig.from_pretrained(pretrained_model_name_or_path, options);
  }
}

/// Device-specific configuration options.
/// @typedef {Omit<TransformersJSConfig, "device" | "device_config">} DeviceConfig
class DeviceConfig {
  /// The data type of the key-value cache.
  /// Either [TensorDataType] or [Map<DataType, TensorDataType>].
  dynamic kv_cache_dtype;

  /// Override the free dimensions of the model.
  /// See https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html#freedimensionoverrides
  /// for more information.
  Map<String, int>? free_dimension_overrides;

  /// The default data type to use for the model.
  /// Either [DataType] or [Map<String, DataType>].
  dynamic dtype;

  /// Whether to load the model using the external data format (used for models >= 2GB in size).
  /// Has a type of [int], [bool], [Map<String, int or bool>]. Check out [ExternalData] for more info.
  dynamic use_external_data_format;

  DeviceConfig({
    this.kv_cache_dtype,
    this.free_dimension_overrides,
    this.dtype,
    this.use_external_data_format,
  });

  factory DeviceConfig.fromJson(Map<String, dynamic> json) => DeviceConfig(
    kv_cache_dtype: json['kv_cache_dtype'],
    free_dimension_overrides: json['free_dimension_overrides'],
    dtype: json['dtype'] is String
        ? DataType.fromString(json['dtype'])
        : (json['dtype'] as Map<String, dynamic>).map((k, v) => MapEntry(k, DataType.fromString(v))),
    use_external_data_format: json['use_external_data_format'],
  );

  Map<String, dynamic> toJson() => {
    'kv_cache_dtype': kv_cache_dtype,
    'free_dimension_overrides': free_dimension_overrides,
    'dtype': dtype,
    'use_external_data_format': use_external_data_format,
  };

  @override
  String toString() => jsonEncode(toJson());
}

/// Transformers.js-specific configuration, possibly present in config.json under the key `transformers.js_config`.
/// @typedef {Object} TransformersJSConfig
/// @property {Record<import('./utils/devices.js').DeviceType, DeviceConfig>} [device_config] Device-specific configurations.
/// @property {import('./utils/tensor.js').DataType|Record<import('./utils/dtypes.js').DataType, import('./utils/tensor.js').DataType>} [kv_cache_dtype] The data type of the key-value cache.
/// @property {Record<string, number>} [free_dimension_overrides] Override the free dimensions of the model.
/// See https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html#freedimensionoverrides
/// for more information.
/// @property {import('./utils/devices.js').DeviceType} [device] The default device to use for the model.
/// @property {import('./utils/dtypes.js').DataType|Record<string, import('./utils/dtypes.js').DataType>} [dtype] The default data type to use for the model.
/// @property {import('./utils/hub.js').ExternalData|Record<string, import('./utils/hub.js').ExternalData>} [use_external_data_format=false] Whether to load the model using the external data format (used for models >= 2GB in size).
class TransformersJSConfig {
  /// Device-specific configurations.
  Map<DeviceType, DeviceConfig>? device_config;

  /// The data type of the key-value cache.
  /// Either [TensorDataType] or [Map<DataType, TensorDataType>].
  dynamic kv_cache_dtype;

  /// Override the free dimensions of the model.
  /// See https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html#freedimensionoverrides
  /// for more information.
  Map<String, int>? free_dimension_overrides;

  /// The default data type to use for the model.
  /// Either [DataType] or [Map<String, DataType>].
  dynamic dtype;

  /// The default device to use for the model.
  DeviceType? device;

  /// Whether to load the model using the external data format (used for models >= 2GB in size).
  /// Has a type of [int], [bool], [Map<String, int or bool>]. Check out [ExternalData] for more info.
  dynamic use_external_data_format;

  TransformersJSConfig({
    this.kv_cache_dtype,
    this.free_dimension_overrides,
    this.dtype,
    this.use_external_data_format,
    this.device_config,
    this.device,
  });

  factory TransformersJSConfig.fromJson(Map<String, dynamic> json) => TransformersJSConfig(
    kv_cache_dtype: json['kv_cache_dtype'] == null
        ? null : json['kv_cache_dtype'] is String
          ? TensorDataType.fromString(json['kv_cache_dtype'])
          : (json['kv_cache_dtype'] as Map).map((k, v) => MapEntry(
      DataType.fromString(k),
      TensorDataType.fromString(v),
      )),
    free_dimension_overrides: json['free_dimension_overrides'],
    dtype: json['dtype'] == null
        ? null : json['dtype'] is String
          ? DataType.fromString(json['dtype'])
          : (json['dtype'] as Map<String, dynamic>).map((k, v) => MapEntry(k, DataType.fromString(v))),
    use_external_data_format: json['use_external_data_format'],
    device_config: json['device_config'] == null
        ? null
        : (json['device_config'] as Map<String, dynamic>).map((k, v) => MapEntry(
      DeviceType.fromString(k),
      DeviceConfig.fromJson(v),
    )),
    device: json['device'],
  );

  Map<String, dynamic> toJson() => {
    'kv_cache_dtype': kv_cache_dtype,
    'free_dimension_overrides': free_dimension_overrides,
    'dtype': dtype,
    'use_external_data_format': use_external_data_format,
    'device_config': device_config,
    'device': device,
  };

  @override
  String toString() => jsonEncode(toJson());
}
