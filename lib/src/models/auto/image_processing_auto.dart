import 'package:transformers/src/base/image_processing_utils.dart';
import 'package:transformers/src/models/image_processors.dart';
import 'package:transformers/src/utils/constants.dart';
import 'package:transformers/src/utils/hub.dart';

class AutoImageProcessor {

  /// @type {typeof ImageProcessor.from_pretrained}
  static Future<ImageProcessor> from_pretrained(String pretrained_model_name_or_path, [PretrainedOptions? options]) async {
    final preprocessorConfig = await getModelJSON(pretrained_model_name_or_path, IMAGE_PROCESSOR_NAME, true, options);

    // Determine image processor class
    final String? key = preprocessorConfig['image_processor_type'] ?? preprocessorConfig['feature_extractor_type'];
    var image_processor_class = AllImageProcessors[key];

    if (image_processor_class == null) {
      if (key != null) {
        // Only log a warning if the class is not found and the key is set.
        // console.warn
        print("Image processor type '$key' not found, assuming base ImageProcessor. Please report this at $GITHUB_ISSUE_URL.");
      }
      image_processor_class = (Map<String, dynamic> config) => ImageProcessor(config);
    }

    // Instantiate image processor
    return image_processor_class(preprocessorConfig);
  }
}
