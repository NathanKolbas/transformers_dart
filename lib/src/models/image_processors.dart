import 'package:transformers/src/base/image_processing_utils.dart';
import 'package:transformers/src/models/idefics3/image_processing_idefics3.dart';

final Map<
    String,
    ImageProcessor Function(Map<String, dynamic> config)
> AllImageProcessors = {
  'Idefics3ImageProcessor': (Map<String, dynamic> config) => Idefics3ImageProcessor(config),
};
