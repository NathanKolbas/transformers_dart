import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:path/path.dart' as path;
import 'package:transformers/src/utils/core.dart';
import 'package:transformers/src/utils/hub.dart';

// Defined here: https://github.com/python-pillow/Pillow/blob/a405e8406b83f8bfb8916e93971edc7407b8b1ff/src/libImaging/Imaging.h#L262-L268
const Map<int, String> RESAMPLING_MAPPING = {
  0: 'nearest',
  1: 'lanczos',
  2: 'bilinear',
  3: 'bicubic',
  4: 'box',
  5: 'hamming',
};

class RawImage {
  /// The pixel data as a view. Note: this is pixel data not the output data for
  /// a file.
  Uint8List data;

  /// The width of the image
  int width;

  /// The height of the image
  int height;

  /// {1|2|3|4} The number of channels
  int channels;

  late img.Image _image;

  img.Image get image => _image;

  /// Create a new `RawImage` object.
  RawImage._(this.data, this.width, this.height, this.channels);

  /// Returns the size of the image (width, height).
  /// @returns {[number, number]} The size of the image (width, height).
  (int, int) get size => (width, height);

  static RawImage fromImage(img.Image image) => RawImage._(
    image.toUint8List(),
    image.width,
    image.height,
    image.numChannels,
  )
    .._image = image.clone();

  /// Helper method for reading an image from a variety of input types.
  /// @param {RawImage|string|URL|Blob|HTMLCanvasElement|OffscreenCanvas} input
  /// @returns The image object.
  ///
  /// **Example:** Read image from a URL.
  /// ```javascript
  /// let image = await RawImage.read('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg');
  /// // RawImage {
  /// //   "data": Uint8ClampedArray [ 25, 25, 25, 19, 19, 19, ... ],
  /// //   "width": 800,
  /// //   "height": 533,
  /// //   "channels": 3
  /// // }
  /// ```
  static Future<RawImage> read(dynamic input) async {
    if (input is RawImage) {
      return input;
    } else if (input is String || input is Uri) {
      return await fromUrl(input);
    } else if (input is Uint8List) {
      return await fromUint8List(input);
    } else {
      throw ArgumentError('Unsupported input type: $input');
    }
  }

  /// Read an image from a URL or file path.
  /// @param {string|URL} url The URL or file path to read the image from.
  /// @returns {Promise<RawImage>} The image object.
  static Future<RawImage> fromUrl(dynamic url) async {
    if (url is! String && url is! Uri) {
      throw ArgumentError('Unsupported input type. Must be a String or Uri');
    }

    final Uint8List data = await getFile(url);
    return fromUint8List(data);
  }

  /// Helper method to create a new Image from a blob.
  /// @param {Blob} blob The blob to read the image from.
  /// @returns {Promise<RawImage>} The image object.
  static Future<RawImage> fromUint8List(Uint8List data) async {
    final image = img.decodeImage(data);
    if (image == null) throw ArgumentError('Invalid image data');

    return RawImage._(image.toUint8List(), image.width, image.height, image.numChannels)
      .._image = image;
  }

  /// Helper method to create a new Image from a tensor
  /// @param {Tensor} tensor
  static fromTensor() {
    // TODO: implement fromTensor
    throw UnimplementedError('TODO implement fromTensor');
  }

  RawImage grayscale() {
    if (channels == 1) {
      return this;
    }

    _image = img.grayscale(_image).convert(numChannels: 1);
    data = _image.toUint8List();
    channels = _image.numChannels;
    return this;
  }

  /// Convert the image to RGB format.
  /// @returns {RawImage} `this` to support chaining.
  RawImage rgb() {
    if (channels == 3) {
      return this;
    }

    switch (channels) {
      case 1: // grayscale to rgb
        final rgb = img.Image.from(_image);
        for (int y = 0; y < _image.height; y++) {
          for (int x = 0; x < _image.width; x++) {
            final g = img.getLuminance(_image.getPixel(x, y));
            rgb.setPixelRgb(x, y, g, g, g);
          }
        }
        _image = rgb;
        break;
      case 4: // rgba to rgb
        final rgb = img.Image.from(_image);
        for (int y = 0; y < _image.height; y++) {
          for (int x = 0; x < _image.width; x++) {
            final c = _image.getPixel(x, y);
            rgb.setPixelRgb(x, y, c.r, c.g, c.b);
          }
        }
        _image = rgb;
        break;
      default:
        throw StateError('Conversion failed due to unsupported number of channels: $channels');
    }

    _image = _image.convert(numChannels: 3);
    data = _image.toUint8List();
    channels = _image.numChannels;
    return this;
  }

  /// Convert the image to RGBA format.
  /// @returns {RawImage} `this` to support chaining.
  RawImage rgba() {
    if (channels == 4) {
      return this;
    }

    switch (channels) {
      case 1: // grayscale to rgba
        final rgba = img.Image.from(_image);
        for (int y = 0; y < _image.height; y++) {
          for (int x = 0; x < _image.width; x++) {
            final g = img.getLuminance(_image.getPixel(x, y));
            rgba.setPixelRgba(x, y, g, g, g, 255);
          }
        }
        _image = rgba;
        break;
      case 3: // rgb to rgba
        final rgba = img.Image.from(_image);
        for (int y = 0; y < _image.height; y++) {
          for (int x = 0; x < _image.width; x++) {
            final c = _image.getPixel(x, y);
            rgba.setPixelRgba(x, y, c.r, c.g, c.b, 255);
          }
        }
        _image = rgba;
        break;
      default:
        throw StateError('Conversion failed due to unsupported number of channels: $channels');
    }

    _image = _image.convert(numChannels: 4);
    data = _image.toUint8List();
    channels = _image.numChannels;
    return this;
  }

  /// Apply an alpha mask to the image. Operates in place.
  /// @param {RawImage} mask The mask to apply. It should have a single channel.
  /// @returns {RawImage} The masked image.
  /// @throws {Error} If the mask is not the same size as the image.
  /// @throws {Error} If the image does not have 4 channels.
  /// @throws {Error} If the mask is not a single channel.
  RawImage putAlpha(RawImage mask) {
    if (mask.width != width || mask.height != height) {
      throw ArgumentError('Expected mask size to be ${width}x$height, but got ${mask.width}x${mask.height}');
    }
    if (mask.channels != 1) {
      throw ArgumentError('Expected mask to have 1 channel, but got ${mask.channels}');
    }

    final output = img.Image(width: _image.width, height: _image.height, numChannels: 4);

    for (int y = 0; y < _image.height; y++) {
      for (int x = 0; x < _image.width; x++) {
        final basePixel = _image.getPixel(x, y);
        final maskPixel = mask._image.getPixel(x, y);

        // Get grayscale value from mask to use as alpha (0–255)
        final alpha = maskPixel.luminance == 0 ? 0 : 255;

        // Preserve RGB from base, apply alpha from mask
        output.setPixelRgba(
          x,
          y,
          basePixel.r,
          basePixel.g,
          basePixel.b,
          alpha,
        );
      }
    }

    _image = output;
    data = output.toUint8List();
    channels = output.numChannels;
    return this;
  }

  /// Resize the image to the given dimensions. This method uses the canvas API to perform the resizing.
  /// @param {number} width The width of the new image. `null` or `-1` will preserve the aspect ratio.
  /// @param {number} height The height of the new image. `null` or `-1` will preserve the aspect ratio.
  /// @param {Object} options Additional options for resizing.
  /// @param {0|1|2|3|4|5|string} [options.resample] The resampling method to use.
  /// @returns {Promise<RawImage>} `this` to support chaining.
  RawImage resize(int? width, int? height, { dynamic resample = 2 }) {
    // Do nothing if the image already has the desired size
    if (this.width == width && this.height == height) {
      return this;
    }
    if (resample is! String && resample is! int && resample is! img.Interpolation) {
      throw ArgumentError('Unsupported resample type. It must be a String, int, or Interpolation.');
    }

    // Calculate width / height to maintain aspect ratio, in the event that
    // the user passed a null value in.
    // This allows users to pass in something like `resize(320, null)` to
    // resize to 320 width, but maintain aspect ratio.
    bool nullish_width = isNullishDimension(width);
    bool nullish_height = isNullishDimension(height);
    if (nullish_width && nullish_height) {
      return this;
    } else if (nullish_width) {
      width = ((height! / this.height) * this.width).floor();
    } else if (nullish_height) {
      height = ((width! / this.width) * this.height).floor();
    }

    final img.Interpolation interpolation;

    if (resample is img.Interpolation) {
      interpolation = resample;
    } else {
      // Ensure resample method is a string
      String resampleMethod = RESAMPLING_MAPPING[resample] ?? resample;

      switch (resampleMethod) {
        case 'box':
          interpolation = img.Interpolation.average;
          break;
        case 'hamming':
        // console.warn
          print('Resampling method hamming is not yet supported. Using box instead.');
          interpolation = img.Interpolation.average;
          break;
        case 'nearest':
          interpolation = img.Interpolation.nearest;
          break;
        case 'bilinear':
          interpolation = img.Interpolation.linear;
          break;
        case 'bicubic':
          interpolation = img.Interpolation.cubic;
          break;
        case 'lanczos':
          print(
            'Warning: "lanczos" resampling is not yet supported by the image '
            'package. Falling back to "bicubic" which is the next best '
            'alternative. It is currently being worked on and you can check the '
            'progress or help out here: '
            'https://github.com/brendan-duncan/image/pull/734',
          );
          // interpolation = img.Interpolation.lanczos;
          interpolation = img.Interpolation.cubic;
          break;
        default:
          throw ArgumentError('Resampling method $resampleMethod is not supported.');
      }
    }


    _image = img.copyResize(
      _image,
      width: width,
      height: height,
      interpolation: interpolation,
    );

    this.width = _image.width;
    this.height = _image.height;
    data = _image.toUint8List();
    channels = _image.numChannels;
    return this;
  }

  RawImage pad(int left, int right, int top, int bottom) {
    left = math.max(left, 0);
    right = math.max(right, 0);
    top = math.max(top, 0);
    bottom = math.max(bottom, 0);

    if (left == 0 && right == 0 && top == 0 && bottom == 0) {
      // No padding needed
      return this;
    }

    final padded = img.Image(
      width: _image.width + left + right,
      height: _image.height + top + bottom,
      numChannels: _image.numChannels,
      // In JS, toSharp().extend defaults to black and so does the image library
      // backgroundColor: img.ColorRgb8(0, 0, 0),
    );

    _image = img.compositeImage(padded, _image, dstX: left, dstY: top);

    width = _image.width;
    height = _image.height;
    data = _image.toUint8List();
    channels = _image.numChannels;
    return this;
  }

  RawImage crop(int x_min, int y_min, int x_max, int y_max) {
    // Note for others, I removed things like adding/subtracting 1 as I don't
    // think that was correct for the image library being used here. If you are
    // experiencing slight differences then this might be a good place to
    // investigate.

    // Ensure crop bounds are within the image
    x_min = math.max(x_min, 0);
    y_min = math.max(y_min, 0);
    x_max = math.min(x_max, width - 1);
    y_max = math.min(y_max, height - 1);

    // Do nothing if the crop is the entire image
    if (x_min == 0 && y_min == 0 && x_max == width - 1 && y_max == height - 1) {
      return this;
    }

    final crop_width = x_max - x_min + 1;
    final crop_height = y_max - y_min + 1;

    _image = img.copyCrop(_image, x: x_min, y: y_min, width: crop_width, height: crop_height);

    width = _image.width;
    height = _image.height;
    data = _image.toUint8List();
    channels = _image.numChannels;
    return this;
  }

  RawImage center_crop(int crop_width, int crop_height) {
    // If the image is already the desired size, return it
    if (width == crop_width && height == crop_height) {
      return this;
    }

    // Determine bounds of the image in the new canvas
    final width_offset = (width - crop_width) / 2;
    final height_offset = (height - crop_height) / 2;

    if (width_offset >= 0 && height_offset >= 0) {
      // Cropped image lies entirely within the original image
      _image = img.copyCrop(
        _image,
        x: width_offset.floor(),
        y: height_offset.floor(),
        width: crop_width,
        height: crop_height,
      );
    } else if (width_offset <= 0 && height_offset <= 0) {
      // Cropped image lies entirely outside the original image,
      // so we add padding
      final top = (-height_offset).floor();
      final left = (-width_offset).floor();
      pad(
        left,
        crop_width - width - left,
        top,
        crop_height - height - top,
      );
    } else {
      // Cropped image lies partially outside the original image.
      // We first pad, then crop.

      final y_padding = [0, 0];
      int y_extract = 0;
      if (height_offset < 0) {
        y_padding[0] = (-height_offset).floor();
        y_padding[1] = crop_height - height - y_padding[0];
      } else {
        y_extract = height_offset.floor();
      }

      final x_padding = [0, 0];
      int x_extract = 0;
      if (width_offset < 0) {
        x_padding[0] = (-width_offset).floor();
        x_padding[1] = crop_width - width - x_padding[0];
      } else {
        x_extract = width_offset.floor();
      }

      pad(
        x_padding[0],
        x_padding[1],
        y_padding[0],
        y_padding[1],
      );
      _image = img.copyCrop(
        _image,
        x: x_extract,
        y: y_extract,
        width: crop_width,
        height: crop_height,
      );
    }

    width = _image.width;
    height = _image.height;
    data = _image.toUint8List();
    channels = _image.numChannels;
    return this;
  }

  /// Returns a copy, not a view, of [data]
  Uint8List toUint8List() => Uint8List.fromList(data);

  /// Split this image into individual bands. This method returns an array of individual image bands from an image.
  /// For example, splitting an "RGB" image creates three new images each containing a copy of one of the original bands (red, green, blue).
  ///
  /// Inspired by PIL's `Image.split()` [function](https://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.Image.split).
  /// @returns {RawImage[]} An array containing bands.
  List<RawImage> split() {
    final channels = _image.numChannels;
    final width = _image.width;
    final height = _image.height;

    // Create empty grayscale images for each channel
    final red = img.Image(width: width, height: height);
    final green = img.Image(width: width, height: height);
    final blue = img.Image(width: width, height: height);
    img.Image? alpha;

    if (channels == 4) {
      alpha = img.Image(width: width, height: height);
    }

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixel = _image.getPixel(x, y);
        final r = pixel.r;
        final g = pixel.g;
        final b = pixel.b;
        final a = pixel.a;

        // Set each band as grayscale
        red.setPixelRgb(x, y, r, r, r);
        green.setPixelRgb(x, y, g, g, g);
        blue.setPixelRgb(x, y, b, b, b);

        if (alpha != null) {
          alpha.setPixelRgb(x, y, a, a, a);
        }
      }
    }

    final List<img.Image> bands = [red, green, blue];
    if (alpha != null) {
      bands.add(alpha);
    }

    return bands.map(fromImage).toList();
  }

  /// Clone the image
  /// @returns {RawImage} The cloned image
  RawImage clone() => fromImage(_image.clone());

  /// Helper method for converting image to have a certain number of channels
  /// @param {number} numChannels The number of channels. Must be 1, 3, or 4.
  /// @returns {RawImage} `this` to support chaining.
  RawImage convert(int numChannels) {
    if (channels == numChannels) return this; // Already correct number of channels

    switch (numChannels) {
      case 1:
        grayscale();
        break;
      case 3:
        rgb();
        break;
      case 4:
        rgba();
        break;
      default:
        throw ArgumentError('Conversion failed due to unsupported number of channels: $channels');
    }
    return this;
  }

  /// Save the image to the given path.
  /// @param {string} path The path to save the image to.
  Future<void> save(String filePath) async {
    final encoder = img.findEncoderForNamedImage(path.basename(filePath)) ?? img.PngEncoder();
    final data = encoder.encode(_image);
    final file = File(filePath);
    await file.create(recursive: true);
    await file.writeAsBytes(data);
  }
}

/// Helper function to load an image from a URL, path, etc.
final load_image = RawImage.read;
