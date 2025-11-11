import 'package:transformers/src/base/processing_utils.dart';
import 'package:transformers/src/models/feature_extractors.dart';
import 'package:transformers/src/models/image_processors.dart';
import 'package:transformers/src/models/processors.dart';
import 'package:transformers/src/utils/constants.dart';
import 'package:transformers/src/utils/hub.dart';

/// Helper class which is used to instantiate pretrained processors with the `from_pretrained` function.
/// The chosen processor class is determined by the type specified in the processor config.
///
/// **Example:** Load a processor using `from_pretrained`.
/// ```javascript
/// let processor = await AutoProcessor.from_pretrained('openai/whisper-tiny.en');
/// ```
///
/// **Example:** Run an image through a processor.
/// ```javascript
/// let processor = await AutoProcessor.from_pretrained('Xenova/clip-vit-base-patch16');
/// let image = await RawImage.read('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg');
/// let image_inputs = await processor(image);
/// // {
/// //   "pixel_values": {
/// //     "dims": [ 1, 3, 224, 224 ],
/// //     "type": "float32",
/// //     "data": Float32Array [ -1.558687686920166, -1.558687686920166, -1.5440893173217773, ... ],
/// //     "size": 150528
/// //   },
/// //   "original_sizes": [
/// //     [ 533, 800 ]
/// //   ],
/// //   "reshaped_input_sizes": [
/// //     [ 224, 224 ]
/// //   ]
/// // }
/// ```
class AutoProcessor {
  static Future<Processor> from_pretrained(
    String pretrained_model_name_or_path,
    [ProcessorPretrainedOptions? options]
  ) async {
    // TODO: first check for processor.json
    final preprocessorConfig = await getModelJSON(
      pretrained_model_name_or_path,
      IMAGE_PROCESSOR_NAME,
      true,
      options,
    );

    final String? imageProcessorType = preprocessorConfig['image_processor_type'];
    final String? featureExtractorType = preprocessorConfig['feature_extractor_type'];
    final String? processorClass = preprocessorConfig['processor_class'];
    if (processorClass != null && AllProcessorsFromPretrained.containsKey(processorClass)) {
      final fromPretrained = AllProcessorsFromPretrained[processorClass]!;
      return await fromPretrained(pretrained_model_name_or_path, options);
    }

    if (imageProcessorType == null && featureExtractorType == null) {
      throw ArgumentError('No `image_processor_type` or `feature_extractor_type` found in the config.');
    }

    final Map<String, dynamic> components = {};
    if (imageProcessorType != null) {
      final imageProcessorClass = AllImageProcessors[imageProcessorType];
      if (imageProcessorClass == null) {
        throw ArgumentError("Unknown image_processor_type: '$imageProcessorType'.");
      }
      components['image_processor'] = imageProcessorClass(preprocessorConfig);
    }

    if (featureExtractorType != null) {
      final imageProcessorClass = AllImageProcessors[featureExtractorType];
      if (imageProcessorClass != null) {
        // Handle legacy case where image processors were specified as feature extractors
        components['image_processor'] = imageProcessorClass(preprocessorConfig);
      } else {
        final featureExtractorClass = AllFeatureExtractors[featureExtractorType];
        if (featureExtractorClass == null) {
          throw ArgumentError("Unknown feature_extractor_type: '$featureExtractorClass'.");
        }
        components['feature_extractor'] = featureExtractorClass(preprocessorConfig);
      }
    }

    final Map<String, dynamic> config = {};
    return Processor(config, components);
  }
}
