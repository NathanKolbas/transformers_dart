import 'dart:math' as math;

/// Helper method to permute a `AnyTypedArray` directly
/// @template {AnyTypedArray} T
/// @param {T} array
/// @param {number[]} dims
/// @param {number[]} axes
/// @returns {[T, number[]]} The permuted array and the new shape.
(List<T>, List<int>) permute_data<T>(List<T> array, List<int> dims, List<int> axes) {
  // Calculate the new shape of the permuted array
  // and the stride of the original array
  final shape = List.filled(axes.length, 0);
  final stride = List.filled(axes.length, 0);

  for (int i=axes.length - 1, s = 1; i >= 0; --i) {
    stride[i] = s;
    shape[i] = dims[axes[i]];
    s *= shape[i];
  }

  // Precompute inverse mapping of stride
  final invStride = axes
      .indexed
      .map((e) => stride[axes.indexOf(e.$1)])
      .toList();

  // Create the permuted array with the new shape
  final permutedData = array.toList();

  // Permute the original array to the new array
  for (int i=0; i < array.length; ++i) {
    int newIndex = 0;
    for (int j=dims.length - 1, k = i; j >= 0; --j) {
      newIndex += (k % dims[j]) * invStride[j];
      k = (k / dims[j]).floor();
    }
    permutedData[newIndex] = array[i];
  }

  return (permutedData, shape);
}

/// Compute the softmax of an array of numbers.
/// @template {TypedArray|number[]} T
/// @param {T} arr The array of numbers to compute the softmax of.
/// @returns {T} The softmax array.
List<double> softmax<T extends num>(List<T> arr) {
  // Compute the maximum value in the array
  final maxVal = max(arr).$1;

  // Compute the exponentials of the array values
  final exps = arr.map((x) => math.exp(x - maxVal));

  // Compute the sum of the exponentials
  final sumExps = exps.fold(0.0, (acc, val) => acc + val);

  // Compute the softmax values
  final softmaxArr = exps.map((x) => x / sumExps);

  return softmaxArr.toList();
}

/// Returns the value and index of the minimum element in an array.
/// @template {number[]|bigint[]|AnyTypedArray} T
/// @param {T} arr array of numbers.
/// @returns {T extends bigint[]|BigTypedArray ? [bigint, number] : [number, number]} the value and index of the minimum element, of the form: [valueOfMin, indexOfMin]
/// @throws {Error} If array is empty.
(T, int) min<T extends num>(List<T> arr) {
  if (arr.isEmpty) throw ArgumentError('Array must not be empty');

  T min = arr[0];
  int indexOfMin = 0;
  for (int i = 1; i < arr.length; ++i) {
    if (arr[i] < min) {
      min = arr[i];
      indexOfMin = i;
    }
  }
  return (min, indexOfMin);
}

/// Returns the value and index of the maximum element in an array.
/// @template {number[]|bigint[]|AnyTypedArray} T
/// @param {T} arr array of numbers.
/// @returns {T extends bigint[]|BigTypedArray ? [bigint, number] : [number, number]} the value and index of the maximum element, of the form: [valueOfMax, indexOfMax]
/// @throws {Error} If array is empty.
(T, int) max<T extends num>(List<T> arr) {
  if (arr.isEmpty) throw ArgumentError('Array must not be empty');

  T max = arr[0];
  int indexOfMax = 0;
  for (int i = 1; i < arr.length; ++i) {
    if (arr[i] > max) {
      max = arr[i];
      indexOfMax = i;
    }
  }
  return (max, indexOfMax);
}

/// Helper function to round a number to the nearest integer, with ties rounded to the nearest even number.
/// Also known as "bankers' rounding". This is the default rounding mode in python. For example:
/// 1.5 rounds to 2 and 2.5 rounds to 2.
///
/// @param {number} x The number to round
/// @returns {number} The rounded number
int bankers_round(double x) {
  final r = x.round();
  final br = x.abs() % 1 == 0.5 ? (r % 2 == 0 ? r : r - 1) : r;
  return br;
}
