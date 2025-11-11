import 'dart:convert';
import 'dart:math' as math;

import 'package:transformers/src/utils/constants.dart';
import 'package:transformers/src/utils/core.dart';
import 'package:transformers/src/utils/hub.dart';
import 'package:transformers/src/utils/image.dart';
import 'package:transformers/src/utils/maths.dart';
import 'package:transformers/src/utils/tensor.dart';

class Size {
  /// The width of the image.
  final int width;

  /// The height of the image.
  final int height;

  Size({
    required this.width,
    required this.height,
  });

  static Size fromJson(Map<String, dynamic> json) => Size(
    width: json['width'],
    height: json['height'],
  );
}

/// @typedef {object} ImageProcessorResult
/// @property {Tensor} pixel_values The pixel values of the batched preprocessed images.
/// @property {HeightWidth[]} original_sizes Array of two-dimensional tuples like [[480, 640]].
/// @property {HeightWidth[]} reshaped_input_sizes Array of two-dimensional tuples like [[1000, 1330]].
class ImageProcessorResult {
  /// The pixel values of the batched preprocessed images.
  final Tensor<double> pixel_values;

  /// Array of two-dimensional tuples like `[[480, 640]]`.
  final List<(int, int)> original_sizes;

  /// Array of two-dimensional tuples like `[[1000, 1330]]`.
  final List<(int, int)> reshaped_input_sizes;

  final Tensor<bool>? pixel_attention_mask;

  final List<List<int>>? rows;

  final List<List<int>>? cols;

  ImageProcessorResult({
    required this.pixel_values,
    required this.original_sizes,
    required this.reshaped_input_sizes,
    this.pixel_attention_mask,
    this.rows,
    this.cols,
  });

  Map<String, dynamic> toJson() => {
    'pixel_values': pixel_values,
    'original_sizes': original_sizes,
    'reshaped_input_sizes': reshaped_input_sizes,
    'pixel_attention_mask': pixel_attention_mask,
    'rows': rows,
    'cols': cols,
  };

  @override
  String toString() => jsonEncode(this);
}

/// Helper function to constrain a value to be a multiple of a number.
/// @param {number} val The value to constrain.
/// @param {number} multiple The number to constrain to.
/// @param {number} [minVal=0] The minimum value to constrain to.
/// @param {number} [maxVal=null] The maximum value to constrain to.
/// @returns {number} The constrained value.
/// @private
int constraint_to_multiple_of(double val, int multiple, [int minVal = 0, int? maxVal]) {
  final a = val / multiple;
  int x = bankers_round(a) * multiple;

  if (maxVal != null && x > maxVal) {
    x = a.floor() * multiple;
  }

  if (x < minVal) {
    x = a.ceil() * multiple;
  }

  return x;
}

/// Rounds the height and width down to the closest multiple of size_divisibility
/// @param {[number, number]} size The size of the image
/// @param {number} divisor The divisor to use.
/// @returns {[number, number]} The rounded size.
(int, int) enforce_size_divisibility((int, int) size, int divisor) {
  final (width, height) = size;
  return (
    math.max((width / divisor).floor(), 1) * divisor,
    math.max((height / divisor).floor(), 1) * divisor,
  );
}

/// Rescales the image so that the following conditions are met:
///
/// 1. Both dimensions (height and width) are divisible by 'factor'.
/// 2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
/// 3. The aspect ratio of the image is maintained as closely as possible.
///
/// @param {number} height The height of the image.
/// @param {number} width The width of the image.
/// @param {number} [factor=28] The factor to use for resizing.
/// @param {number} [min_pixels=56*56] The minimum number of pixels.
/// @param {number} [max_pixels=14*14*4*1280] The maximum number of pixels.
/// @returns {[number, number]} The new height and width of the image.
/// @throws {Error} If the height or width is smaller than the factor.
(int, int) smart_resize(int height, int width, [int factor = 28, int min_pixels = 56 * 56, int max_pixels = 14 * 14 * 4 * 1280]) {
  if (height < factor || width < factor) {
    throw ArgumentError('height:$height or width:$width must be larger than factor:$factor');
  } else if (math.max(height, width) / math.min(height, width) > 200) {
  throw ArgumentError(
    'absolute aspect ratio must be smaller than 200, got ${math.max(height, width) / math.min(height, width)}'
  );
  }

  int h_bar = (height / factor).round() * factor;
  int w_bar = (width / factor).round() * factor;

  if (h_bar * w_bar > max_pixels) {
    final beta = math.sqrt((height * width) / max_pixels);
    h_bar = ((height / beta) / factor).floor() * factor;
    w_bar = ((width / beta) / factor).floor() * factor;
  } else if (h_bar * w_bar < min_pixels) {
    final beta = math.sqrt(min_pixels / (height * width));
    h_bar = ((height * beta) / factor).ceil() * factor;
    w_bar = ((width * beta) / factor).ceil() * factor;
  }

  return (h_bar, w_bar);
}

class ImageProcessorConfig {
  Map<String, dynamic> json;

  /// {function} [progress_callback=null] If specified, this function will be called during model construction, to provide the user with progress updates.
  Function? progress_callback;

  /// {number[]} [image_mean] The mean values for image normalization.
  List<double>? image_mean;

  /// {number[]} [image_std] The standard deviation values for image normalization.
  List<double>? image_std;

  /// {boolean} [do_rescale] Whether to rescale the image pixel values to the [0,1] range.
  bool? do_rescale;

  /// {number} [rescale_factor] The factor to use for rescaling the image pixel values.
  double? rescale_factor;

  /// {boolean} [do_normalize] Whether to normalize the image pixel values.
  bool? do_normalize;

  /// {boolean} [do_resize] Whether to resize the image.
  bool? do_resize;

  /// {number} [resample] What method to use for resampling.
  int? resample;

  /// {number|Object} [size] The size to resize the image to.
  dynamic size;

  /// {number|Object} [image_size] The size to resize the image to (same as `size`).
  dynamic image_size;

  /// {boolean} [do_flip_channel_order=false] Whether to flip the color channels from RGB to BGR.
  /// Can be overridden by the `do_flip_channel_order` parameter in the `preprocess` method.
  bool do_flip_channel_order;

  /// {boolean} [do_center_crop] Whether to center crop the image to the specified `crop_size`.
  /// an be overridden by `do_center_crop` in the `preprocess` method.
  bool? do_center_crop;

  /// {boolean} [do_thumbnail] Whether to resize the image using thumbnail method.
  bool? do_thumbnail;

  /// {boolean} [keep_aspect_ratio] If `true`, the image is resized to the largest possible size such that the aspect ratio is preserved.
  /// Can be overridden by `keep_aspect_ratio` in `preprocess`.
  bool? keep_aspect_ratio;

  /// {number} [ensure_multiple_of] If `do_resize` is `true`, the image is resized to a size that is a multiple of this value.
  /// Can be overridden by `ensure_multiple_of` in `preprocess`.
  int? ensure_multiple_of;

  /// {number[]} [mean] The mean values for image normalization (same as `image_mean`).
  List<double>? mean;

  /// {number[]} [std] The standard deviation values for image normalization (same as `image_std`).
  List<double>? std;


  ImageProcessorConfig({
    this.json = const {},
    this.progress_callback,
    this.image_mean,
    this.image_std,
    this.do_rescale,
    this.rescale_factor,
    this.do_normalize,
    this.do_resize,
    this.resample,
    this.size,
    this.image_size,
    bool? do_flip_channel_order,
    this.do_center_crop,
    this.do_thumbnail,
    this.keep_aspect_ratio,
    this.ensure_multiple_of,
    this.mean,
    this.std,
  }) : do_flip_channel_order = do_flip_channel_order ?? false;

  static ImageProcessorConfig fromJson(Map<String, dynamic> json) => ImageProcessorConfig(
    json: json,
    progress_callback: json['progress_callback'],
    image_mean: json['image_mean'] != null ? List<double>.from(json['image_mean']) : null,
    image_std: json['image_std'] != null ? List<double>.from(json['image_std']) : null,
    do_rescale: json['do_rescale'],
    rescale_factor: json['rescale_factor'],
    do_normalize: json['do_normalize'],
    do_resize: json['do_resize'],
    resample: json['resample'],
    size: json['size'],
    image_size: json['image_size'],
    do_flip_channel_order: json['do_flip_channel_order'],
    do_center_crop: json['do_center_crop'],
    do_thumbnail: json['do_thumbnail'],
    keep_aspect_ratio: json['keep_aspect_ratio'],
    ensure_multiple_of: json['ensure_multiple_of'],
    mean: json['mean'] != null ? List<double>.from(json['mean']) : null,
    std: json['std'] != null ? List<double>.from(json['std']) : null,
  );
}

/// @typedef {object} PreprocessedImage
/// @property {HeightWidth} original_size The original size of the image.
/// @property {HeightWidth} reshaped_input_size The reshaped input size of the image.
/// @property {Tensor} pixel_values The pixel values of the preprocessed image.
class PreprocessedImage {
  /// The original size of the image.
  (int, int) original_size;

  /// The reshaped input size of the image.
  (int, int) reshaped_input_size;

  /// The pixel values of the preprocessed image.
  Tensor<double> pixel_values;

  PreprocessedImage({
    required this.original_size,
    required this.reshaped_input_size,
    required this.pixel_values,
  });
}

class ImageProcessor {
  /// The configuration object.
  late final ImageProcessorConfig config;

  dynamic image_mean;
  dynamic image_std;
  dynamic resample;
  dynamic do_rescale;
  dynamic rescale_factor;
  late bool do_normalize;
  late bool do_thumbnail;
  dynamic size;
  late bool do_resize;
  dynamic size_divisibility;
  late bool do_center_crop;
  dynamic crop_size;
  late bool do_convert_rgb;
  late bool do_crop_margin;
  dynamic pad_size;
  late bool do_pad;
  dynamic min_pixels;
  dynamic max_pixels;
  late bool do_flip_channel_order;

  /// Constructs a new `ImageProcessor`.
  ImageProcessor(Map<String, dynamic> config) {
    image_mean = config['image_mean'] ?? config['mean'];
    image_std = config['image_std'] ?? config['std'];

    resample = config['resample'] ?? 2; // 2 => bilinear
    do_rescale = config['do_rescale'] ?? true;
    rescale_factor = config['rescale_factor'] ?? (1 / 255);
    do_normalize = config['do_normalize'] ?? false;

    do_thumbnail = config['do_thumbnail'] ?? false;
    size = config['size'] ?? config['image_size'];
    do_resize = config['do_resize'] ?? (size != null);
    size_divisibility = config['size_divisibility'] ?? config['size_divisor'];

    do_center_crop = config['do_center_crop'] ?? false;
    crop_size = config['crop_size'];
    do_convert_rgb = config['do_convert_rgb'] ?? true;
    do_crop_margin = config['do_crop_margin'] ?? false;

    pad_size = config['pad_size'];
    do_pad = config['do_pad'] ?? false;
    min_pixels = config['min_pixels'];
    max_pixels = config['max_pixels'];

    if (do_pad && pad_size != null && size != null && size.width != null && size.height != null) {
      // Should pad, but no pad size specified
      // We infer the pad size from the resize size
      pad_size = size;
    }

    do_flip_channel_order = config['do_flip_channel_order'] ?? false;

    this.config = ImageProcessorConfig.fromJson(config);
  }

  /// Resize the image to make a thumbnail. The image is resized so that no dimension is larger than any
  /// corresponding dimension of the specified size.
  /// @param {RawImage} image The image to be resized.
  /// @param {{height:number, width:number}} size The size `{"height": h, "width": w}` to resize the image to.
  /// @param {string | 0 | 1 | 2 | 3 | 4 | 5} [resample=2] The resampling filter to use.
  /// @returns {Promise<RawImage>} The resized image.
  RawImage thumbnail(RawImage image, Size size, [int resample = 2]) {
    final input_height = image.height;
    final input_width = image.width;

    final output_height = size.height;
    final output_width = size.width;

    // We always resize to the smallest of either the input or output size.
    int height = math.min(input_height, output_height);
    int width = math.min(input_width, output_width);

    if (height == input_height && width == input_width) {
      return image;
    }
    if (input_height > input_width) {
      width = (input_width * height / input_height).floor();
    } else if (input_width > input_height) {
      height = (input_height * width / input_width).floor();
    }
    return image.resize(width, height, resample: resample);
  }

  /// Crops the margin of the image. Gray pixels are considered margin (i.e., pixels with a value below the threshold).
  /// @param {RawImage} image The image to be cropped.
  /// @param {number} gray_threshold Value below which pixels are considered to be gray.
  /// @returns {Promise<RawImage>} The cropped image.
  RawImage crop_margin(RawImage image, [int gray_threshold = 200]) {
    final gray_image = image.clone().grayscale();

    final minValue = gray_image.data.reduce(math.min);
    final maxValue = gray_image.data.reduce(math.max);
    final diff = maxValue - minValue;

    if (diff == 0) {
      return image;
    }

    final threshold = gray_threshold / 255;

    int x_min = gray_image.width, y_min = gray_image.height, x_max = 0, y_max = 0;
    final gray_image_data = gray_image.data;
    for (int j = 0; j < gray_image.height; ++j) {
      final row = j * gray_image.width;
      for (int i = 0; i < gray_image.width; ++i) {
        if ((gray_image_data[row + i] - minValue) / diff < threshold) {
          // We have a non-zero pixel, so we update the min/max values accordingly
          x_min = math.min(x_min, i);
          y_min = math.min(y_min, j);
          x_max = math.max(x_max, i);
          y_max = math.max(y_max, j);
        }
      }
    }

    return image.crop(x_min, y_min, x_max, y_max);
  }

  /// Pad the image by a certain amount.
  /// @param {Float32Array} pixelData The pixel data to pad.
  /// @param {number[]} imgDims The dimensions of the image (height, width, channels).
  /// @param {{width:number; height:number}|number|'square'} padSize The dimensions of the padded image.
  /// @param {Object} options The options for padding.
  /// @param {'constant'|'symmetric'} [options.mode='constant'] The type of padding to add.
  /// @param {boolean} [options.center=false] Whether to center the image.
  /// @param {number|number[]} [options.constant_values=0] The constant value to use for padding.
  /// @returns {[Float32Array, number[]]} The padded pixel data and image dimensions.
  (List<double> , List<int>) pad_image(List<double> pixelData, List<int> imgDims, dynamic padSize, {
    String mode = 'constant',
    bool center = false,
    dynamic constant_values = 0.0,
  }) {
    final [imageHeight, imageWidth, imageChannels] = imgDims;

    final int paddedImageWidth, paddedImageHeight;
    if (padSize is int) {
      paddedImageWidth = padSize;
      paddedImageHeight = padSize;
    } else if (padSize == 'square') {
      paddedImageWidth = paddedImageHeight = math.max(imageHeight, imageWidth);
    } else {
      paddedImageWidth = padSize.width;
      paddedImageHeight = padSize.height;
    }

    // Only add padding if there is a difference in size
    if (paddedImageWidth != imageWidth || paddedImageHeight != imageHeight) {
      List<double> paddedPixelData = [];
      if (constant_values is List) {
        paddedPixelData = List<double>.filled(paddedImageWidth * paddedImageHeight * imageChannels, 0.0);
        // Fill with constant values, cycling through the array
        for (int i = 0; i < paddedPixelData.length; ++i) {
          paddedPixelData[i] = constant_values[i % imageChannels];
        }
      } else if (constant_values != 0) {
        paddedPixelData = List<double>.filled(paddedImageWidth * paddedImageHeight * imageChannels, constant_values);
      }

      final [left, top] = center
          ? [((paddedImageWidth - imageWidth) / 2).floor(), ((paddedImageHeight - imageHeight) / 2).floor()]
          : [0, 0];

      // Copy the original image into the padded image
      for (int i = 0; i < imageHeight; ++i) {
        final a = (i + top) * paddedImageWidth;
        final b = i * imageWidth;
        for (int j = 0; j < imageWidth; ++j) {
          final c = (a + j + left) * imageChannels;
          final d = (b + j) * imageChannels;
          for (int k = 0; k < imageChannels; ++k) {
            paddedPixelData[c + k] = pixelData[d + k];
          }
        }
      }

      if (mode == 'symmetric') {
        if (center) {
          throw ArgumentError('`center` padding is not supported when `mode` is set to `symmetric`.');
          // TODO: Implement this
        }
        final h1 = imageHeight - 1;
        final w1 = imageWidth - 1;
        for (int i = 0; i < paddedImageHeight; ++i) {
          final a = i * paddedImageWidth;
          final b = calculateReflectOffset(i, h1) * imageWidth;

          for (int j = 0; j < paddedImageWidth; ++j) {
            if (i < imageHeight && j < imageWidth) continue; // Do not overwrite original image
            final c = (a + j) * imageChannels;
            final d = (b + calculateReflectOffset(j, w1)) * imageChannels;

            // Copy channel-wise
            for (int k = 0; k < imageChannels; ++k) {
              paddedPixelData[c + k] = pixelData[d + k];
            }
          }
        }
      }


      // Update pixel data and image dimensions
      pixelData = paddedPixelData;
      imgDims = [paddedImageHeight, paddedImageWidth, imageChannels];
    }
    return (pixelData, imgDims);
  }

  /// Rescale the image' pixel values by `this.rescale_factor`.
  /// @param {Float32Array} pixelData The pixel data to rescale.
  /// @returns {void}
  void rescale(List<double> pixelData) {
    for (int i = 0; i < pixelData.length; ++i) {
      pixelData[i] = rescale_factor * pixelData[i];
    }
  }

  /// Find the target (width, height) dimension of the output image after
  /// resizing given the input image and the desired size.
  /// @param {RawImage} image The image to resize.
  /// @param {any} size The size to use for resizing the image.
  /// @returns {[number, number]} The target (width, height) dimension of the output image after resizing.
  (int, int) get_resize_output_image_size(RawImage image, dynamic size) {
    // `size` comes in many forms, so we need to handle them all here:
    // 1. `size` is an integer, in which case we resize the image to be a square

    final (srcWidth, srcHeight) = image.size;

    int? shortest_edge;
    int? longest_edge;

    if (do_thumbnail) {
      // NOTE: custom logic for `Donut` models
      size = Map<String, int>.from(size);
      final height = size['height'];
      final width = size['width'];
      shortest_edge = math.min(height, width);
    }
    // Support both formats for backwards compatibility
    else if (size is int) {
      shortest_edge = size;
      longest_edge = config.json['max_size'] ?? shortest_edge;

    } else if (size != null) {
      // Extract known properties from `size`
      size = Map<String, int>.from(size);
      shortest_edge = size['shortest_edge'];
      longest_edge = size['longest_edge'];
    }

    // If `longest_edge` and `shortest_edge` are set, maintain aspect ratio and resize to `shortest_edge`
    // while keeping the largest dimension <= `longest_edge`
    if (shortest_edge != null || longest_edge != null) {
      // http://opensourcehacker.com/2011/12/01/calculate-aspect-ratio-conserving-resize-for-images-in-javascript/
      // Try resize so that shortest edge is `shortest_edge` (target)
      final shortResizeFactor = shortest_edge == null
        ? 1 // If `shortest_edge` is not set, don't upscale
        : math.max(shortest_edge / srcWidth, shortest_edge / srcHeight);

      final newWidth = srcWidth * shortResizeFactor;
      final newHeight = srcHeight * shortResizeFactor;

      // The new width and height might be greater than `longest_edge`, so
      // we downscale again to ensure the largest dimension is `longest_edge`
      final longResizeFactor = longest_edge == null
          ? 1 // If `longest_edge` is not set, don't downscale
          : math.min(longest_edge / newWidth, longest_edge / newHeight);

      // To avoid certain floating point precision issues, we round to 2 decimal places
      // TODO: Check if this is even needed in dart
      int finalWidth = (double.parse((newWidth * longResizeFactor).toStringAsFixed(2))).floor();
      int finalHeight = (double.parse((newHeight * longResizeFactor).toStringAsFixed(2))).floor();

      if (size_divisibility != null) {
        (finalWidth, finalHeight) = enforce_size_divisibility((finalWidth, finalHeight), size_divisibility);
      }

      return (finalWidth, finalHeight);
    } else if (size != null && size.width != null && size.height != null) {
      // If `width` and `height` are set, resize to those dimensions

      int newWidth = size.width;
      int newHeight = size.height;

      // Custom for DPT models
      final ensure_multiple_of = config.ensure_multiple_of;
      if (ensure_multiple_of == true && ensure_multiple_of != null && ensure_multiple_of != 0) {

        // determine new height and width
        double scale_height = newHeight / srcHeight;
        double scale_width = newWidth / srcWidth;

        // scale as little as possible
        if ((1 - scale_width).abs() < (1 - scale_height).abs()) {
          // fit width
          scale_height = scale_width;
        } else {
          // fit height
          scale_width = scale_height;
        }

        newHeight = constraint_to_multiple_of(scale_height * srcHeight, ensure_multiple_of);
        newWidth = constraint_to_multiple_of(scale_width * srcWidth, ensure_multiple_of);
      }

      return (newWidth, newHeight);
    } else if (size_divisibility != null) {
      return enforce_size_divisibility((srcWidth, srcHeight), size_divisibility);
    } else if (min_pixels != null && max_pixels != null) {
      // Custom resize logic for Qwen2-VL models
      final factor = config.json['patch_size'] * config.json['merge_size'];
      return smart_resize(srcHeight, srcWidth, factor, min_pixels!, max_pixels!);
    } else {
      throw StateError('Could not resize image due to unsupported `this.size` option in config: $size');
    }
  }

  /// Resizes the image.
  /// @param {RawImage} image The image to resize.
  /// @returns {Promise<RawImage>} The resized image.
  RawImage resize(RawImage image) {
    final (newWidth, newHeight) = get_resize_output_image_size(image, size);
    return image.resize(newWidth, newHeight, resample: resample);
  }

  /// Preprocesses the given image.
  ///
  /// @param {RawImage} image The image to preprocess.
  /// @param {Object} overrides The overrides for the preprocessing options.
  /// @returns {Promise<PreprocessedImage>} The preprocessed image.
  Future<PreprocessedImage> preprocess(RawImage image, {
    bool? do_normalize,
    bool? do_pad,
    bool? do_convert_rgb,
    bool? do_convert_grayscale,
    bool? do_flip_channel_order,
  }) async {
    if (do_crop_margin) {
      // NOTE: Specific to nougat processors. This is done before resizing,
      // and can be interpreted as a pre-preprocessing step.
      image = crop_margin(image);
    }

    final (srcWidth, srcHeight) = image.size; // original image size

    // Convert image to RGB if specified in config.
    if (do_convert_rgb ?? this.do_convert_rgb) {
      image = image.rgb();
    } else if (do_convert_grayscale == true) {
      image = image.grayscale();
    }

    // TODO:
    // For efficiency reasons, it might be best to merge the resize and center crop operations into one.

    // Resize all images
    if (do_resize) {
      image = resize(image);
    }

    // Resize the image using thumbnail method.
    if (do_thumbnail) {
      image = thumbnail(image, size, resample);
    }

    if (do_center_crop) {
      int crop_width;
      int crop_height;
      if (crop_size is int) {
        crop_width = crop_size;
        crop_height = crop_size;
      } else {
        final crop_size = Map<String, int>.from(this.crop_size);
        crop_width = crop_size['width']!;
        crop_height = crop_size['height']!;
      }

      image = image.center_crop(crop_width, crop_height);
    }

    /** @type {HeightWidth} */
    final reshaped_input_size = (image.height, image.width);

    // NOTE: All pixel-level manipulation (i.e., modifying `pixelData`)
    // occurs with data in the hwc format (height, width, channels),
    // to emulate the behavior of the original Python code (w/ numpy).
    /** @type {Float32Array} */
    List<double> pixelData = image.toUint8List().map((x) => x.toDouble()).toList();
    List<int> imgDims = [image.height, image.width, image.channels];

    if (do_rescale) {
      rescale(pixelData);
    }

    if (do_normalize ?? this.do_normalize) {
      final List<num> image_mean = this.image_mean is! List
          ? List.filled(image.channels, this.image_mean)
          : List.from(this.image_mean);

      final List<num> image_std = this.image_std is! List
          // TODO: I am pretty sure this is a bug in transformers.js as it used image_mean and not image_std:
          // image_std = new Array(image.channels).fill(image_mean);
          ? List.filled(image.channels, this.image_std)
          : List.from(this.image_std);

      if (image_mean.length != image.channels || image_std.length != image.channels) {
        throw StateError('When set to arrays, the length of `image_mean` (${image_mean.length}) and `image_std` (${image_std.length}) must match the number of channels in the image (${image.channels}).');
      }

      for (int i = 0; i < pixelData.length; i += image.channels) {
        for (int j = 0; j < image.channels; ++j) {
          pixelData[i + j] = (pixelData[i + j] - image_mean[j]) / image_std[j];
        }
      }
    }

    // do padding after rescaling/normalizing
    if (do_pad ?? this.do_pad) {
      if (pad_size != null) {
        final padded = pad_image(pixelData, [image.height, image.width, image.channels], pad_size);
        (pixelData, imgDims) = padded; // Update pixel data and image dimensions
      } else if (size_divisibility != null) {
        final (paddedWidth, paddedHeight) = enforce_size_divisibility((imgDims[1], imgDims[0]), size_divisibility);
        (pixelData, imgDims) = pad_image(pixelData, imgDims, { 'width': paddedWidth, 'height': paddedHeight });
      }
    }

    if (do_flip_channel_order ?? this.do_flip_channel_order) {
      if (imgDims[2] != 3) {
        throw StateError('Flipping channel order is only supported for RGB images.');
      }
      // Convert RGB to BGR
      for (int i = 0; i < pixelData.length; i += 3) {
        final temp = pixelData[i];
        pixelData[i] = pixelData[i + 2];
        pixelData[i + 2] = temp;
      }
    }

    Tensor<double> pixel_values = await Tensor.create(TensorDataType.float32, pixelData, imgDims);
    pixel_values = await pixel_values.permute([2, 0, 1]); // convert to channel dimension format (hwc -> chw)

    return PreprocessedImage(
      original_size: (srcHeight, srcWidth),
      reshaped_input_size: reshaped_input_size,
      pixel_values: pixel_values,
    );
  }

  /// Calls the feature extraction process on an array of images,
  /// preprocesses each image, and concatenates the resulting
  /// features into a single Tensor.
  /// @param {RawImage[]} images The image(s) to extract features from.
  /// @param {...any} args Additional arguments.
  /// @returns {Promise<ImageProcessorResult>} An object containing the concatenated pixel values (and other metadata) of the preprocessed images.
  Future<ImageProcessorResult> call(dynamic images, [List<dynamic> args = const []]) async {
    if (images is! List){
      images = [images];
    }

    final List<PreprocessedImage> imageData = await Future.wait(images.map((x) => preprocess(x)));

    // Stack pixel values
    final pixel_values = await stack(imageData.map((x) => x.pixel_values).toList(), 0);

    return ImageProcessorResult(
      pixel_values: pixel_values,

      // Original sizes of images
      original_sizes: imageData.map((x) => x.original_size).toList(),

      // Reshaped sizes of images, before padding or cropping
      reshaped_input_sizes: imageData.map((x) => x.reshaped_input_size).toList(),
    );
  }

  /// Instantiate one of the processor classes of the library from a pretrained model.
  ///
  /// The processor class to instantiate is selected based on the `image_processor_type` (or `feature_extractor_type`; legacy)
  /// property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible)
  ///
  /// @param {string} pretrained_model_name_or_path The name or path of the pretrained model. Can be either:
  /// - A string, the *model id* of a pretrained processor hosted inside a model repo on huggingface.co.
  ///   Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
  ///   user or organization name, like `dbmdz/bert-base-german-cased`.
  /// - A path to a *directory* containing processor files, e.g., `./my_model_directory/`.
  /// @param {import('../utils/hub.js').PretrainedOptions} options Additional options for loading the processor.
  ///
  /// @returns {Promise<ImageProcessor>} A new instance of the Processor class.
  static Future<ImageProcessor> from_pretrained(String pretrained_model_name_or_path, [PretrainedOptions? options]) async {
    final preprocessorConfig = await getModelJSON(pretrained_model_name_or_path, IMAGE_PROCESSOR_NAME, true, options);
    return ImageProcessor(preprocessorConfig);
  }
}
