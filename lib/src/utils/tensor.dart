/// @file Helper module for `Tensor` processing.
///
/// These functions and classes are only used internally,
/// meaning an end-user shouldn't need to access anything here.
///
/// @module utils/tensor

import 'dart:math' as math;

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:transformers/src/backends/onnx.dart' as onnx;
import 'package:transformers/src/ops/registry.dart';
import 'package:transformers/src/utils/maths.dart';


/// The data type for a [Tensor]
enum TensorDataType {
  float64,
  float32,
  float16,
  bfloat16,
  string,
  int8,
  uint8,
  int16,
  uint16,
  int32,
  uint32,
  int64,
  uint64,
  bool,
  uint4,
  int4;

  static TensorDataType fromString(String dataType) => TensorDataType
      .values
      .firstWhere((e) => e.name == dataType);
}

class Tensor<T> {
  /// @type {number[]} Dimensions of the tensor.
  List<int> get dims => ort_tensor.dims;

  // set dims(value) {
  //   // FIXME: ONNXTensor declares dims as readonly so one needs to use the constructor() if dims change.
  //   ort_tensor.dims = value;
  // }

  /// @type {DataType} Type of the tensor.
  TensorDataType get type => ort_tensor.type;

  /// @type {DataArray} The data stored in the tensor.
  List<T> get data => ort_tensor.data;

  /// @type {number} The number of elements in the tensor.
  int get size => ort_tensor.size;

  /// @type {string} The location of the tensor data.
  String get location => ort_tensor.location;

  onnx.Tensor<T> ort_tensor;

  /// Create a new Tensor or copy an existing Tensor.
  /// @param {[DataType, DataArray, number[]]|[ONNXTensor]} args
  Tensor._(this.ort_tensor);

  static Future<Tensor<T>> create<T>(
    TensorDataType tensorDataType,
    List<T> data,
    [List<int>? shape]
  ) async {
    final ort_tensor = await onnx.Tensor.create<T>(
      dataType: tensorDataType,
      data: data,
      shape: shape,
    );

    return Tensor<T>._(ort_tensor);
  }

  static Future<Tensor<T>> createFromOrtValue<T>(OrtValue ortValue) async {
    final ort_tensor = await onnx.Tensor.createFromOrtValue<T>(ortValue);

    return Tensor<T>._(ort_tensor);
  }

  /// Used to update the data in-place. Directly modifying the [data] list will
  /// have no effect since it is transferred outside of dart.
  Future<Tensor<T>> updateData(List<T> data) async {
    await ort_tensor.updateData(data);
    return this;
  }

  Future<void> dispose() async {
    await ort_tensor.dispose();
  }

  @override
  String toString() => 'Tensor { data: $data, type: $type, dims: $dims, size: $size }';

  Future<Tensor<T>> operator [](int index) async => await _getitem(index);

  /// Index into a Tensor object.
  /// @param {number} index The index to access.
  /// @returns {Tensor} The data at the specified index.
  Future<Tensor<T>> _getitem(int index) async {
    final [iterLength, ...iterDims] = dims;

    index = safeIndex(index, iterLength);

    if (iterDims.isNotEmpty) {
      final iterSize = iterDims.reduce((a, b) => a * b);
      return _subarray(index, iterSize, iterDims);
    } else {
      return await Tensor.create(type, [data[index]], iterDims);
    }
  }

  /// Casts the generic [T] of this class to a new type. Ideal for updating a
  /// [Tensor<dynamic>] to it's known type.
  ///
  /// Note: This does not convert the data to a new type - only updating dart's
  /// generic type for this class.
  ///
  /// Important: The returned [Tensor] points to the same underlying data and is
  /// not copied into a new Tensor.
  Tensor<U> cast<U>() => Tensor._(ort_tensor.cast<U>());

  /// Casts this tensor to a new type.
  ///
  /// This creates a new tensor with the same shape but with the data
  /// converted to the specified type `U`.
  ///
  /// Note: The previous Tensor is *NOT* disposed.
  ///
  /// @param {Type} U The target data type (e.g., `int`, `double`).
  /// @returns {Future<Tensor<U>>} A new tensor with the casted data.
  /// @throws {ArgumentError} If the cast is between unsupported types.
  // Future<Tensor<U>> cast<U>() async {
  //   // Determine the new TensorDataType based on the target type U.
  //   final TensorDataType newDataType;
  //   if (U == double) {
  //     newDataType = TensorDataType.float32;
  //   } else if (U == int) {
  //     newDataType = TensorDataType.int64;
  //   } else if (U == bool) {
  //     newDataType = TensorDataType.bool;
  //   } else {
  //     throw ArgumentError('Unsupported cast type: $U');
  //   }
  //
  //   // Create the new data list by casting each element.
  //   final List<U> newData;
  //   final currentData = this.data;
  //
  //   if (U == double) {
  //     if (currentData is! List<num>) {
  //       throw StateError('Cannot cast from non-numeric type $T to double.');
  //     }
  //
  //     newData = (currentData as List<num>).map((e) => e.toDouble()).toList() as List<U>;
  //   } else if (U == int) {
  //     if (currentData is! List<num>) {
  //       throw StateError('Cannot cast from non-numeric type $T to int.');
  //     }
  //
  //     newData = (currentData as List<num>).map((e) => e.toInt()).toList() as List<U>;
  //   } else {
  //     // For other types, a more direct cast might work, but can be unsafe.
  //     // This example focuses on numeric conversions.
  //     try {
  //       newData = currentData.cast<U>();
  //     } on TypeError {
  //       throw ArgumentError('Failed to cast from $T to $U.');
  //     }
  //   }
  //
  //   // Create and return the new tensor with the casted data.
  //   return await Tensor.create<U>(newDataType, newData, dims);
  // }

  /// @param {number} index
  /// @param {number} iterSize
  /// @param {any} iterDims
  /// @returns {Tensor}
  Future<Tensor<T>> _subarray(int index, int iterSize, List<int> iterDims) async {
    final o1 = index * iterSize;
    final o2 = (index + 1) * iterSize;

    // We use subarray if available (typed array), otherwise we use slice (normal array)
    final data = this.data.sublist(o1, o2);
    return await Tensor.create(type, data, iterDims);
  }

  /// Returns the value of this tensor as a standard JavaScript Number. This only works
  /// for tensors with one element. For other cases, see `Tensor.tolist()`.
  /// @returns {number|bigint} The value of this tensor as a standard JavaScript Number.
  /// @throws {Error} If the tensor has more than one element.
  T item() {
    final this_data = data;
    if (this_data.length != 1) {
      throw StateError('a Tensor with ${this_data.length} elements cannot be converted to Scalar');
    }
    return this_data.first;
  }

  /// Convert tensor data to a n-dimensional JS list
  /// @returns {Array}
  List<dynamic> tolist() {
    return reshape(data, dims);
  }

  /// Return a new Tensor with the sigmoid function applied to each element.
  /// @returns {Tensor} The tensor with the sigmoid function applied.
  Future<Tensor<double>> sigmoid() async {
    final List<T> this_data = data;
    if (T != num) throw StateError('This tensor is not a num');

    final List<double> newData = List.generate(
      this_data.length,
          (i) => T is double
          ? this_data[i] as double
          : (this_data[i] as num).toDouble(),
    );

    for (int i=0; i < newData.length; ++i) {
      newData[i] = 1 / (1 + math.exp(-newData[i]));
    }

    await dispose();

    return await create<double>(type, newData, dims);
  }

  /// Return a new Tensor with a callback function applied to each element.
  /// @param {Function} callback - The function to apply to each element. It should take three arguments:
  ///                              the current element, its index, and the tensor's data array.
  /// @returns {Tensor} A new Tensor with the callback function applied to each element.
  Future<Tensor<U>> map<U>(U Function(T element, int index, List<T> data) callback) async {
    final List<U> newData = data
        .indexed
        .map((e) => callback(e.$2, e.$1, data))
        .toList();

    await dispose();

    return await create<U>(type, newData, dims);
  }

  /// Return a new Tensor with every element multiplied by a constant.
  /// @param {number} val The value to multiply by.
  /// @returns {Tensor} The new tensor.
  Future<Tensor<T>> mul(num val) async {
    if (T != num) throw StateError('This tensor is not a num');

    final List<T> newData = data
        .map((e) => (e as num) * val)
        .cast<T>()
        .toList();

    await dispose();

    return await create<T>(type, newData, dims);
  }

  /// Return a new Tensor with every element divided by a constant.
  /// @param {number} val The value to divide by.
  /// @returns {Tensor} The new tensor.
  Future<Tensor<T>> div(num val) async {
    if (T != num) throw StateError('This tensor is not a num');

    final List<T> newData = data
        .map((e) => (e as num) / val)
        .cast<T>()
        .toList();

    await dispose();

    return await create<T>(type, newData, dims);
  }

  /// Return a new Tensor with every element added by a constant.
  /// @param {number} val The value to add by.
  /// @returns {Tensor} The new tensor.
  Future<Tensor<T>> add(num val) async {
    if (T != num) throw StateError('This tensor is not a num');

    final List<T> newData = data
        .map((e) => (e as num) + val)
        .cast<T>()
        .toList();

    await dispose();

    return await create<T>(type, newData, dims);
  }

  /// Return a new Tensor with every element subtracted by a constant.
  /// @param {number} val The value to subtract by.
  /// @returns {Tensor} The new tensor.
  Future<Tensor<T>> sub(num val) async {
    if (T != num) throw StateError('This tensor is not a num');

    final List<T> newData = data
        .map((e) => (e as num) - val)
        .cast<T>()
        .toList();

    await dispose();

    return await create<T>(type, newData, dims);
  }

  /// Creates a deep copy of the current Tensor.
  /// @returns {Tensor} A new Tensor with the same type, data, and dimensions as the original.
  Future<Tensor<T>> clone() async {
    return await create<T>(type, data.toList(), dims);
  }

  /// Performs a slice operation on the Tensor along specified dimensions.
  ///
  /// Consider a Tensor that has a dimension of [4, 7]:
  /// ```
  /// [ 1,  2,  3,  4,  5,  6,  7]
  /// [ 8,  9, 10, 11, 12, 13, 14]
  /// [15, 16, 17, 18, 19, 20, 21]
  /// [22, 23, 24, 25, 26, 27, 28]
  /// ```
  /// We can slice against the two dims of row and column, for instance in this
  /// case we can start at the second element, and return to the second last,
  /// like this:
  /// ```
  /// tensor.slice([1, -1], [1, -1]);
  /// ```
  /// which would return:
  /// ```
  /// [  9, 10, 11, 12, 13 ]
  /// [ 16, 17, 18, 19, 20 ]
  /// ```
  ///
  /// @param {...(number|number[]|null)} slices The slice specifications for each dimension.
  /// - If a number is given, then a single element is selected.
  /// - If an array of two numbers is given, then a range of elements [start, end (exclusive)] is selected.
  /// - If null is given, then the entire dimension is selected.
  /// @returns {Tensor} A new Tensor containing the selected elements.
  /// @throws {Error} If the slice input is invalid.
  Future<Tensor<T>> slice([List<dynamic>? slices]) async {
    // This allows for slicing with ranges and numbers
    final List<int> newTensorDims = [];
    final List<(int, int)> newOffsets = [];

    // slices is an array of numbers or arrays of numbers
    // e.g., slices = [0, [1, 3], null, [0, 3]]
    for (int sliceIndex = 0; sliceIndex < dims.length; ++sliceIndex) {
      dynamic slice = slices == null
          ? null
          : (sliceIndex >= slices.length ? null : slices[sliceIndex]);

      if (slice == null) {
        // null or undefined means take the whole dimension
        newOffsets.add((0, dims[sliceIndex]));
        newTensorDims.add(dims[sliceIndex]);
      } else if (slice is int) {
        slice = safeIndex(slice, dims[sliceIndex], sliceIndex);

        // A number means take a single element
        newOffsets.add((slice, slice + 1));
      } else if (slice is List && slice.length == 2) {
        // An array of length 2 means take a range of elements
        var [int? start, int? end] = slice;
        start = start == null
            ? 0
            : safeIndex(start, dims[sliceIndex], sliceIndex, false);
        end = end == null
            ? dims[sliceIndex]
            : safeIndex(end, dims[sliceIndex], sliceIndex, false);

        if (start > end) {
          throw Exception('Invalid slice: $slice');
        }

        final offsets = (
          math.max(start, 0),
          math.min(end, dims[sliceIndex])
        );

        newOffsets.add(offsets);
        newTensorDims.add(offsets.$2 - offsets.$1);
      } else {
        throw Exception('Invalid slice: $slice');
      }
    }

    final newDims = newOffsets.map((e) => e.$2 - e.$1).toList();
    final newBufferSize = newDims.reduce((a, b) => a * b);

    final this_data = this.data;
    // Allocate memory
    final data = this_data.sublist(0, newBufferSize);

    // Precompute strides
    final stride = this.stride();

    for (int i = 0; i < newBufferSize; ++i) {
      int originalIndex = 0;
      for (int j = newDims.length - 1, num = i; j >= 0; --j) {
        final size = newDims[j];
        originalIndex += ((num % size) + newOffsets[j].$1) * stride[j];
        num = (num / size).floor();
      }
      data[i] = this_data[originalIndex];
    }
    return await Tensor.create(type, data, newTensorDims);
  }

  /// Return a permuted version of this Tensor, according to the provided dimensions.
  /// @param  {...number} dims Dimensions to permute.
  /// @returns {Tensor} The permuted tensor.
  Future<Tensor<T>> permute(List<int> dims) async {
    final (permutedData, shape) = permute_data(data, this.dims, dims);
    await dispose();
    return await create<T>(type, permutedData, shape);
  }

  // TODO: implement transpose. For now (backwards compatibility), it's just an alias for permute()
  Future<Tensor<T>> transpose(List<int> dims) => permute(dims);

  /// Returns the matrix norm or vector norm of a given tensor.
  /// @param {number|string} [p='fro'] The order of norm
  /// @param {number} [dim=null] Specifies which dimension of the tensor to calculate the norm across.
  /// If dim is None, the norm will be calculated across all dimensions of input.
  /// @param {boolean} [keepdim=false] Whether the output tensors have dim retained or not.
  /// @returns {Tensor} The norm of the tensor.
  Future<Tensor> norm([dynamic p = 'fro', int? dim = null, bool keepdim = false]) async {
    if (p == 'fro') {
      // NOTE: Since we only support integer dims, Frobenius norm produces the same result as p=2.
      p = 2;
    } else if (p is String) {
      throw ArgumentError('Unsupported norm: $p');
    }

    final this_data = data;
    fn(num a, T b, [_, _]) => a + (math.pow(b as num, p));

    if (dim == null) {
      final val = math.pow(this_data.fold(0, fn), 1 / p);
      return await create<T>(this.type, [val as T], []);
    }

    final (type, result, resultDims) = reduce_helper<T, num>(fn, this, dim, keepdim);

    if (p != 1) {
      for (int i = 0; i < result.length; ++i) {
        result[i] = math.pow(result[i], 1 / p);
      }
    }
    return await create(type, result, resultDims);
  }

  /// Performs `L_p` normalization of inputs over specified dimension.
  /// @param {number} [p=2] The exponent value in the norm formulation
  /// @param {number} [dim=1] The dimension to reduce
  /// @returns {Tensor} The normalized tensor.
  Future<Tensor> normalize([double p = 2.0, int dim = 1]) async {
    dim = safeIndex(dim, dims.length);

    final norm = await this.norm(p, dim, true);

    final this_data = data.toList();
    final norm_data = norm.data;
    for (int i = 0; i < this_data.length; ++i) {

      // Calculate the index in the resulting array
      int resultIndex = 0;

      for (int j = dims.length - 1, num = i, resultMultiplier = 1; j >= 0; --j) {
        final size = dims[j];
        if (j != dim) {
          final index = num % size;
          resultIndex += index * resultMultiplier;
          resultMultiplier *= dims[j];
        }
        num = (num / size).floor();
      }

      // Divide by normalized value
      this_data[i] = ((this_data[i] as num) / (norm_data[resultIndex] as num)) as T;
    }

    await norm.dispose();
    await dispose();
    return await create<T>(type, this_data, dims);
  }

  /// Compute and return the stride of this tensor.
  /// Stride is the jump necessary to go from one element to the next one in the specified dimension dim.
  /// @returns {number[]} The stride of this tensor.
  List<int> stride() => dimsToStride(dims);

  /// Returns a new tensor with a dimension of size one inserted at the specified position.
  ///
  /// NOTE: The returned tensor shares the same underlying data with this tensor.
  ///
  /// @param {number} dim The index at which to insert the singleton dimension
  /// @returns {Tensor} The unsqueezed tensor
  /// TODO: [dim] seems like it can be null but if it is then it is just an empty array in JS. Verify how this should be handled.
  Future<Tensor<T>> unsqueeze(int dim) async {
    await dispose();
    return await create<T>(
      type,
      data,
      calc_unsqueeze_dims(dims, dim),
    );
  }

  /// Returns a new tensor with the same data as the `self` tensor but of a different `shape`.
  /// @param  {...number} dims the desired size
  /// @returns {Tensor} The tensor with the same data but different shape
  Future<Tensor<T>> view(List<int> dims) async {
    // TODO: validate dims
    int inferredIndex = -1;
    for (int i = 0; i < dims.length; ++i) {
      if (dims[i] == -1) {
        if (inferredIndex != -1) {
          throw ArgumentError("Only one dimension can be inferred");
        }
        inferredIndex = i;
      }
    }

    final this_data = data;
    if (inferredIndex != -1) {
      // Some dimension must be inferred
      int productOther = 1;
      for (int i = 0; i < dims.length; i++) {
        final dim = dims[i];

        if (i != inferredIndex) {
          productOther *= dim;
        }
      }

      dims[inferredIndex] = (this_data.length / productOther).toInt();
    }

    // TODO: There is a not about "uses same underlying storage" is this going to be a problem?
    return Tensor.create(type, this_data, dims); // NOTE: uses same underlying storage
  }
}

/**
 * This creates a nested array of a given type and depth (see examples).
 *
 * @example
 *   NestArray<string, 1>; // string[]
 * @example
 *   NestArray<number, 2>; // number[][]
 * @example
 *   NestArray<string, 3>; // string[][][] etc.
 * @template T
 * @template {number} Depth
 * @template {never[]} [Acc=[]]
 * @typedef {Acc['length'] extends Depth ? T : NestArray<T[], Depth, [...Acc, never]>} NestArray
 */

/// Reshapes a 1-dimensional array into an n-dimensional array, according to the provided dimensions.
///
/// @example
///   reshape([10                    ], [1      ]); // Type: number[]      Value: [10]
///   reshape([1, 2, 3, 4            ], [2, 2   ]); // Type: number[][]    Value: [[1, 2], [3, 4]]
///   reshape([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]); // Type: number[][][]  Value: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
///   reshape([1, 2, 3, 4, 5, 6, 7, 8], [4, 2   ]); // Type: number[][]    Value: [[1, 2], [3, 4], [5, 6], [7, 8]]
/// @param {T[]|DataArray} data The input array to reshape.
/// @param {DIM} dimensions The target shape/dimensions.
/// @template T
/// @template {[number]|number[]} DIM
/// @returns {NestArray<T, DIM["length"]>} The reshaped array.
List<dynamic> reshape<T>(List<T> data, List<int> dimensions) {
  final totalElements = data.length;
  final dimensionSize = dimensions.reduce((a, b) => a * b);

  if (totalElements != dimensionSize) {
    throw ArgumentError('cannot reshape array of size $totalElements into shape ($dimensions)');
  }

  dynamic reshapedArray = data;

  for (int i = dimensions.length - 1; i >= 0; i--) {
    List<dynamic> newArray = [[]];

    for (var val in reshapedArray) {
      List<dynamic> lastArray = newArray.last;

      if (lastArray.length < dimensions[i]) {
        lastArray.add(val);
      } else {
        newArray.add([val]);
      }
    }

    reshapedArray = newArray;
  }

  return reshapedArray[0];
}

enum Interpolate4dMode {
  nearest,
  bilinear,
  bicubic,
}

/// Down/up samples the input.
/// Inspired by https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html.
/// @param {Tensor} input the input tensor
/// @param {Object} options the options for the interpolation
/// @param {[number, number]|[number, number, number]|[number, number, number, number]} [options.size=null] output spatial size.
/// @param {"nearest"|"bilinear"|"bicubic"} [options.mode='bilinear'] algorithm used for upsampling
/// @returns {Promise<Tensor>} The interpolated tensor.
Future<Tensor> interpolate_4d<T>(Tensor<T> input, {
  List<int>? size,
  Interpolate4dMode mode = Interpolate4dMode.bilinear,
}) async {
  // Error checking
  if (input.dims.length != 4) {
    throw ArgumentError('`interpolate_4d` currently only supports 4D input.');
  }
  if (size == null) {
    // TODO: support scale_factor
    throw ArgumentError('`interpolate_4d` requires a `size` argument.');
  }

  // Fill in missing dimensions
  final List<int> targetDims;
  if (size.length == 2) {
    targetDims = [...input.dims.sublist(0, 2), ...size];
  } else if (size.length == 3) {
    targetDims = [input.dims[0], ...size];
  } else if (size.length == 4) {
    targetDims = size;
  } else {
    throw ArgumentError('`size` must be of length 2, 3, or 4.');
  }

  final op = switch (mode) {
    Interpolate4dMode.nearest => await TensorOpRegistry.nearest_interpolate_4d,
    Interpolate4dMode.bilinear => await TensorOpRegistry.bilinear_interpolate_4d,
    Interpolate4dMode.bicubic => await TensorOpRegistry.bicubic_interpolate_4d,
  };

  final sizeTensor = await Tensor.create(
    TensorDataType.int64,
    targetDims,
    [targetDims.length],
  );
  return (await op({ 'x': input, 's': sizeTensor })).first;
}

/// Matrix product of two tensors.
/// Inspired by https://pytorch.org/docs/stable/generated/torch.matmul.html
/// @param {Tensor} a the first tensor to be multiplied
/// @param {Tensor} b the second tensor to be multiplied
/// @returns {Promise<Tensor>} The matrix product of the two tensors.
Future<Tensor> matmul(Tensor a, Tensor b) async {
  final op = await TensorOpRegistry.matmul;
  return (await op({ 'a': a, 'b': b })).first;
}

/// Returns the k largest elements of the given input tensor.
/// Inspired by https://pytorch.org/docs/stable/generated/torch.topk.html
/// @param {Tensor} x the input tensor
/// @param {number} [k] the k in "top-k"
/// @returns {Promise<[Tensor, Tensor]>} the output tuple of (Tensor, LongTensor) of top-k elements and their indices.
Future<(Tensor, Tensor)> topk(Tensor x, [int? k]) async {
  final op = await TensorOpRegistry.top_k;

  if (k == null) {
    k = x.dims.last;
  } else {
    k = math.min(k, x.dims.last);
  }

  final [topK, indicies] = await op({
    'x': x,
    'k': await Tensor.create(
      TensorDataType.int64,
      [k],
      [1],
    ),
  });
  return (topK, indicies);
}

Future<Tensor> arrayToIndexTensor(List<int> array) async => await Tensor.create<int>(
  TensorDataType.int64,
  array,
  [array.length],
);

/// Slice a multidimensional float32 tensor.
/// @param {Tensor} data: Tensor of data to extract slices from
/// @param {number[]} starts: 1-D array of starting indices of corresponding axis in axes
/// @param {number[]} ends: 1-D array of ending indices (exclusive) of corresponding axis in axes
/// @param {number[]} axes: 1-D array of axes that starts and ends apply to
/// @param {number[]} [steps]: 1-D array of slice step of corresponding axis in axes.
/// @returns {Promise<Tensor>} Sliced data tensor.
Future<Tensor> slice(Tensor data, List<int> starts, List<int> ends, List<int> axes, [List<int>? steps]) async {
  final op = await TensorOpRegistry.slice;
  return (await op({
    'x': data,
    's': await arrayToIndexTensor(starts),
    'e': await arrayToIndexTensor(ends),
    'a': await arrayToIndexTensor(axes),
    't': await arrayToIndexTensor(steps ?? List.filled(axes.length, 1)),
  })).first;
}

/// Perform mean pooling of the last hidden state followed by a normalization step.
/// @param {Tensor} last_hidden_state Tensor of shape [batchSize, seqLength, embedDim]
/// @param {Tensor} attention_mask Tensor of shape [batchSize, seqLength]
/// @returns {Tensor} Returns a new Tensor of shape [batchSize, embedDim].
Future<Tensor> mean_pooling(Tensor last_hidden_state, Tensor attention_mask) async {
  // last_hidden_state: [batchSize, seqLength, embedDim]
  // attention_mask:    [batchSize, seqLength]
  final lastHiddenStateData = last_hidden_state.data;
  final attentionMaskData = List<num>.from(attention_mask.data);

  final shape = [last_hidden_state.dims[0], last_hidden_state.dims[2]];

  final returnedData = List<num>.filled(shape[0] * shape[1], 0);
  final [batchSize, seqLength, embedDim] = last_hidden_state.dims;

  int outIndex = 0;
  for (int i = 0; i < batchSize; ++i) {
    final offset = i * embedDim * seqLength;

    for (int k = 0; k < embedDim; ++k) {
      num sum = 0;
      num count = 0;

      final attnMaskOffset = i * seqLength;
      final offset2 = offset + k;
      // Pool over all words in sequence
      for (int j = 0; j < seqLength; ++j) {
        // index into attention mask
        final attn = attentionMaskData[attnMaskOffset + j];

        count += attn;
        sum += lastHiddenStateData[offset2 + j * embedDim] * attn;
      }

      final avg = sum / count;
      returnedData[outIndex++] = avg;
    }
  }

  return await Tensor.create(
    last_hidden_state.type,
    returnedData,
    shape,
  );
}

/// Helper function to calculate new dimensions when performing an unsqueeze operation.
/// @param {number[]} dims The dimensions of the tensor.
/// @param {number} dim The dimension to unsqueeze.
/// @returns {number[]} The new dimensions.
/// @private
List<int> calc_unsqueeze_dims(List<int> dims, int dim) {
  // Dimension out of range (e.g., "expected to be in range of [-4, 3], but got 4")
  // + 1 since we allow inserting at the end (i.e. dim = -1)
  dim = safeIndex(dim, dims.length + 1);
  dims = dims.toList();
  // Insert 1 into specified dimension
  dims.insert(dim, 1);
  return dims;
}

/// Safely calculate the index for an array of a given size, allowing negative indexing.
/// @param {number} index The index that will be used.
/// @param {number} size The size of the array.
/// @param {number} [dimension=null] The dimension that the index is for (optional).
/// @returns {number} The index, guaranteed to be non-negative and less than `arrayLength`.
///
/// @throws {Error} If the index is out of range.
/// @private
int safeIndex(int index, int size, [int? dimension, bool boundsCheck = true]) {
  if (index < -size || index >= size) {
    if (boundsCheck) {
      throw StateError('IndexError: index $index is out of bounds for dimension${dimension == null ? '' : ' $dimension'} with size $size');
    } else {
      return index < -size ? 0 : size;
    }
  }

  if (index < 0) {
    // Negative indexing, ensuring positive index
    index = ((index % size) + size) % size;
  }
  return index;
}

/// Concatenates an array of tensors along a specified dimension.
/// @param {Tensor[]} tensors The array of tensors to concatenate.
/// @param {number} dim The dimension to concatenate along.
/// @returns {Tensor} The concatenated tensor.
Future<Tensor<T>> cat<T>(List<Tensor<T>> tensors, [int dim = 0]) async {
  dim = safeIndex(dim, tensors[0].dims.length);

  // TODO do validation of shapes

  final resultDims = tensors[0].dims.toList();
  resultDims[dim] = tensors.fold(0, (a, b) => a + b.dims[dim]);

  // Create a new array to store the accumulated values
  final resultSize = resultDims.fold(1, (a, b) => a * b);
  List<T> result = [];

  // Create output tensor of same type as first
  final resultType = tensors[0].type;

  if (dim == 0) {
    // Handle special case for performance reasons

    for (final tensor in tensors) {
      final tensorData = tensor.data;
      result.addAll(tensorData);
    }
  } else {
    result = List.filled(resultSize, (T == int ? 0 : 0.0) as T);
    int currentDim = 0;

    for (int t=0; t < tensors.length; ++t) {
      final tensor = tensors[t];
      final data = tensor.data,
          dims = tensor.dims;

      // Iterate over the data array
      for (int i=0; i < data.length; ++i) {
        // Calculate the index in the resulting array
        int resultIndex = 0;

        for (int j=dims.length - 1, num = i, resultMultiplier = 1; j >= 0; --j) {
          final size = dims[j];
          int index = num % size;
          if (j == dim) {
            index += currentDim;
          }
          resultIndex += index * resultMultiplier;
          resultMultiplier *= resultDims[j];
          num = (num / size).floor();
        }
        // Accumulate the value at the current index
        result[resultIndex] = data[i];
      }

      currentDim += dims[dim];
    }
  }

  return await Tensor.create(resultType, result, resultDims);
}
/// Stack an array of tensors along a specified dimension.
/// @param {Tensor[]} tensors The array of tensors to stack.
/// @param {number} dim The dimension to stack along.
/// @returns {Tensor} The stacked tensor.
Future<Tensor<T>> stack<T>(List<Tensor<T>> tensors, [int dim = 0]) async {
  // TODO do validation of shapes
  // NOTE: stack expects each tensor to be equal size
  return await cat(
    await Future.wait(tensors.map((t) => t.unsqueeze(dim))),
    dim,
  );
}

/// @param {(previousValue: any, currentValue: any, currentIndex?: number, resultIndex?: number) => any} callbackfn
/// @param {Tensor} input the input tensor.
/// @param {number|null} dim the dimension to reduce.
/// @param {boolean} keepdim whether the output tensor has dim retained or not.
/// @returns {[DataType, any, number[]]} The reduced tensor data.
(TensorDataType, List<U>, List<int>) reduce_helper<T, U>(
  U Function(U previousValue, T currentValue, [int currentIndex, int resultIndex]) callbackfn,
  Tensor<T> input,
  int dim,
  [bool keepdim = false, U? initialValue]
) {
  final inputData = input.data;
  final inputDims = input.dims;

  // Negative indexing
  dim = safeIndex(dim, inputDims.length);

  // Calculate the shape of the resulting array after summation
  final resultDims = inputDims.toList(); // Copy the original dimensions
  resultDims[dim] = 1; // Remove the specified axis

  // Create a new array to store the accumulated values
  final result = List<U>.filled((inputData.length / inputDims[dim]).floor(), initialValue ?? 0 as U);

  // Iterate over the data array
  for (int i = 0; i < inputData.length; ++i) {
    // Calculate the index in the resulting array
    int resultIndex = 0;

    for (int j = inputDims.length - 1, num = i, resultMultiplier = 1; j >= 0; --j) {
      final size = inputDims[j];
      if (j != dim) {
        final index = num % size;
        resultIndex += index * resultMultiplier;
        resultMultiplier *= resultDims[j];
      }
      num = (num / size).floor();
    }

    // Accumulate the value at the current index
    result[resultIndex] = callbackfn(result[resultIndex], inputData[i], i, resultIndex);
  }

  if (!keepdim) resultDims.removeAt(dim);

  return (input.type, result, resultDims);
}

List<int> dimsToStride(List<int> dims) {
  final stride = List<int>.filled(dims.length, 0);
  for (int i = dims.length - 1, s2 = 1; i >= 0; --i) {
    stride[i] = s2;
    s2 *= dims[i];
  }
  return stride;
}

Future<Tensor<T>> fullHelper<T>(List<int> size, T fill_value, TensorDataType dtype) async {
  final numElements = size.fold(1, (a, b) => a * b);
  return await Tensor.create(
    dtype,
    List.filled(numElements, fill_value),
    size,
  );
}

/// Creates a tensor of size size filled with fill_value. The tensor's dtype is inferred from fill_value.
/// @param {number[]} size A sequence of integers defining the shape of the output tensor.
/// @param {number|bigint|boolean} fill_value The value to fill the output tensor with.
/// @returns {Tensor} The filled tensor.
Future<Tensor<T>> full<T>(List<int> size, T fill_value, [TensorDataType? dtype]) async {
  if (dtype == null) {
    if (fill_value is double) {
      // TODO: In js this is float32. It might make sense to use float64 since this isn't js. This could make things slower though...
      dtype = TensorDataType.float32;
    } else if (fill_value is int) {
      dtype = TensorDataType.int64;
    } else if (fill_value is bool) {
      dtype = TensorDataType.bool;
    } else {
      // TODO: support other dtypes
      throw ArgumentError('Unsupported data type: $T');
    }
  }

  return await fullHelper(size, fill_value, dtype);
}

Future<Tensor<T>> full_like<T>(Tensor<T> tensor, T fill_value, [TensorDataType? dtype]) async {
  return await full(tensor.dims, fill_value, dtype);
}

/// Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size.
/// @param {number[]} size A sequence of integers defining the shape of the output tensor.
/// @returns {Tensor} The ones tensor.
Future<Tensor<int>> ones(List<int> size) async {
  return await fullHelper(size, 1, TensorDataType.int64);
}

/// Returns a tensor filled with the scalar value 1, with the same size as input.
/// @param {Tensor} tensor The size of input will determine size of the output tensor.
/// @returns {Tensor} The ones tensor.
Future<Tensor<int>> ones_like(Tensor tensor) async {
  return await ones(tensor.dims);
}

/// Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.
/// @param {number[]} size A sequence of integers defining the shape of the output tensor.
/// @returns {Tensor} The zeros tensor.
Future<Tensor<int>> zeros(List<int> size) async {
  return await fullHelper(size, 0, TensorDataType.int64);
}

/// Returns a tensor filled with the scalar value 0, with the same size as input.
/// @param {Tensor} tensor The size of input will determine size of the output tensor.
/// @returns {Tensor} The zeros tensor.
Future<Tensor<int>> zeros_like(Tensor tensor) async {
  return await zeros(tensor.dims);
}

enum QuantizeEmbeddingsPrecision { binary, ubinary }

/// Quantizes the embeddings tensor to binary or unsigned binary precision.
/// @param {Tensor} tensor The tensor to quantize.
/// @param {'binary'|'ubinary'} precision The precision to use for quantization.
/// @returns {Tensor} The quantized tensor.
Future<Tensor> quantize_embeddings(Tensor tensor, QuantizeEmbeddingsPrecision precision) async {
  if (tensor.dims.length != 2) {
    throw ArgumentError("The tensor must have 2 dimensions");
  }
  if (tensor.dims.last % 8 != 0) {
    throw ArgumentError("The last dimension of the tensor must be a multiple of 8");
  }

  final signed = precision == QuantizeEmbeddingsPrecision.binary;
  final dtype = signed ? TensorDataType.int8 : TensorDataType.uint8;

  // Create a typed array to store the packed bits
  final inputData = tensor.data;
  final outputData = List<int>.filled((inputData.length / 8).floor(), 0);

  // Iterate over each number in the array
  for (int i = 0; i < inputData.length; ++i) {
    // Determine if the number is greater than 0
    final bit = inputData[i] > 0 ? 1 : 0;

    // Calculate the index in the typed array and the position within the byte
    final arrayIndex = (i / 8).floor();
    final bitPosition = i % 8;

    // Pack the bit into the typed array
    outputData[arrayIndex] |= bit << (7 - bitPosition);
    if (signed && bitPosition == 0) {
      outputData[arrayIndex] -= 128;
    }
  };

  return await Tensor.create(
    dtype,
    outputData,
    [tensor.dims[0], (tensor.dims[1] / 8).floor()],
  );
}
