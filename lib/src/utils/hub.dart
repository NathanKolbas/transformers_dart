// ignore_for_file: non_constant_identifier_names

import 'dart:convert';
import 'dart:io';

import 'package:dio/dio.dart' hide ProgressCallback;
import 'package:flutter/foundation.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:transformers/src/backends/onnx.dart';
import 'package:transformers/src/configs.dart';
import 'package:transformers/src/env.dart';
import 'package:transformers/src/utils/core.dart';
import 'package:huggingface_hub/huggingface_hub.dart';

import 'package:transformers/src/utils/devices.dart';
import 'package:transformers/src/utils/dtypes.dart';

// TODO: Put this somewhere better
Dio dio = Dio();

/// @typedef {boolean|number} ExternalData Whether to load the model using the external data format (used for models >= 2GB in size).
/// If `true`, the model will be loaded using the external data format.
/// If a number, this many chunks will be loaded using the external data format (of the form: "model.onnx_data[_{chunk_number}]").
typedef ExternalData = dynamic;

const int MAX_EXTERNAL_DATA_CHUNKS = 100;

/// Options for loading a pretrained model.
class PretrainedOptions {
  /// If specified, this function will be called during model construction, to provide the user with progress updates.
  ProgressCallback? progress_callback;

  /// Configuration for the model to use instead of an automatically loaded configuration. Configuration can be automatically loaded when:
  /// - The model is a model provided by the library (loaded with the *model id* string of a pretrained model).
  /// - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a configuration JSON file named *config.json* is found in the directory.
  PretrainedConfig? config;

  /// Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used.
  String? cache_dir;

  /// Whether or not to only look at local files (e.g., not try downloading the model).
  bool local_files_only = false;

  /// The specific model version to use. It can be a branch name, a tag name, or a commit id,
  /// since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.
  /// NOTE: This setting is ignored for local requests.
  String revision = 'main';

  PretrainedOptions({
    this.progress_callback,
    this.config,
    this.cache_dir,
    this.local_files_only = false,
    this.revision = 'main',
  });

  factory PretrainedOptions.fromJson(Map<String, dynamic> json) => PretrainedModelOptions(
    progress_callback: json['progress_callback'],
    config: json['config'],
    cache_dir: json['cache_dir'],
    local_files_only: json['local_files_only'] ?? false,
    revision: json['revision'] ?? 'main',
  );

  Map<String, dynamic> toJson() => {
    'progress_callback': progress_callback,
    'config': config,
    'cache_dir': cache_dir,
    'local_files_only': local_files_only,
    'revision': revision,
  };

  @override
  String toString() => jsonEncode(this);
}

/// @typedef {Object} ModelSpecificPretrainedOptions Options for loading a pretrained model.
/// @property {string} [subfolder='onnx'] In case the relevant files are located inside a subfolder of the model repo on huggingface.co,
/// you can specify the folder name here.
/// @property {string} [model_file_name=null] If specified, load the model with this name (excluding the .onnx suffix). Currently only valid for encoder- or decoder-only models.
/// @property {import("./devices.js").DeviceType|Record<string, import("./devices.js").DeviceType>} [device=null] The device to run the model on. If not specified, the device will be chosen from the environment settings.
/// @property {import("./dtypes.js").DataType|Record<string, import("./dtypes.js").DataType>} [dtype=null] The data type to use for the model. If not specified, the data type will be chosen from the environment settings.
/// @property {ExternalData|Record<string, ExternalData>} [use_external_data_format=false] Whether to load the model using the external data format (used for models >= 2GB in size).
/// @property {import('onnxruntime-common').InferenceSession.SessionOptions} [session_options] (Optional) User-specified session options passed to the runtime. If not provided, suitable defaults will be chosen.
class ModelSpecificPretrainedOptions {
  /// In case the relevant files are located inside a subfolder of the model
  /// repo on huggingface.co, you can specify the folder name here.
  String subfolder;

  /// If specified, load the model with this name (excluding the .onnx suffix).
  /// Currently only valid for encoder- or decoder-only models.
  String? model_file_name;

  /// The device to run the model on. If not specified, the device will be chosen from the environment settings.
  dynamic device;

  /// The data type to use for the model. If not specified, the data type will
  /// be chosen from the environment settings.
  dynamic dtype;

  /// Whether to load the model using the external data format (used for models
  /// >= 2GB in size).
  /// [use_external_data_format] has a type of [int], [bool], [Map<String, int or bool>]
  dynamic use_external_data_format;

  /// User-specified session options passed to the runtime. If not provided,
  /// suitable defaults will be chosen.
  OrtSessionOptions? session_options;

  ModelSpecificPretrainedOptions({
    this.subfolder = 'onnx',
    this.model_file_name,
    this.device,
    this.dtype,
    // In the typedef for transformers.js it defaults to false. This makes no
    // sense as otherwise it wouldn't be possible to nullish over to
    // [config.transformersJsConfig.use_external_data_format] if it was not set
    // in the config. Otherwise, how do we know if the config explicitly set to
    // false?
    this.use_external_data_format,
    this.session_options,
  });
}

/// @typedef {PretrainedOptions & ModelSpecificPretrainedOptions} PretrainedModelOptions Options for loading a pretrained model.
class PretrainedModelOptions implements PretrainedOptions, ModelSpecificPretrainedOptions {
  @override
  String? cache_dir;

  @override
  PretrainedConfig? config;

  @override
  bool local_files_only;

  @override
  ProgressCallback? progress_callback;

  @override
  String revision;

  /// [device] may be of the type [DeviceType] or [Map<String, DeviceType>]
  @override
  dynamic device;

  /// [device] may be of the type [DataType] or [Map<String, DataType>]
  @override
  dynamic dtype;

  @override
  String? model_file_name;

  @override
  OrtSessionOptions? session_options;

  @override
  String subfolder;

  /// [use_external_data_format] has a type of [int], [bool], [Map<String, int or bool>]
  @override
  dynamic use_external_data_format;

  PretrainedModelOptions({
    this.progress_callback,
    Map<String, dynamic>? config,
    this.cache_dir,
    this.local_files_only = false,
    this.revision = 'main',
    this.subfolder = 'onnx',
    this.model_file_name,
    dynamic device,
    dynamic dtype,
    // In the typedef for transformers.js it defaults to false. This makes no
    // sense as otherwise it wouldn't be possible to nullish over to
    // [config.transformersJsConfig.use_external_data_format] if it was not set
    // in the config. Otherwise, how do we know if the config explicitly set to
    // false?
    this.use_external_data_format,
    this.session_options,
  }) : config = config == null ? null : PretrainedConfig.fromJson(config),
        device = _parseDevice(device),
        dtype = _parseDtype(dtype);

  static dynamic _parseDevice(dynamic device) {
    if (device == null) return null;
    if (device is DeviceType) return device;
    if (device is Map<String, DeviceType>) return device;

    if (device is String) return DeviceType.fromString(device);
    if (device is Map) return device.map((k, v) => MapEntry(k, DeviceType.fromString(v)));
  }

  static dynamic _parseDtype(dynamic dtype) {
    if (dtype == null) return null;
    if (dtype is DataType) return dtype;
    if (dtype is Map<String, DataType>) return dtype;

    if (dtype is String) return DataType.fromString(dtype);
    if (dtype is Map) return dtype.map((k, v) => MapEntry(k, DataType.fromString(v)));
  }

  factory PretrainedModelOptions.fromJson(Map<String, dynamic> json) => PretrainedModelOptions(
    cache_dir: json['cache_dir'],
    config: json['config'],
    local_files_only: json['local_files_only'] ?? false,
    progress_callback: json['progress_callback'],
    revision: json['revision'] ?? 'main',
    device: json['device'],
    dtype: json['dtype'],
    model_file_name: json['model_file_name'],
    session_options: json['session_options'] == null ? null : OrtSessionOptionsEx.fromJson(json['session_options']),
    subfolder: json['subfolder'] ?? 'onnx',
    use_external_data_format: json['use_external_data_format'],
  );

  @override
  Map<String, dynamic> toJson() => {
    'cache_dir': cache_dir,
    'config': config,
    'local_files_only': local_files_only,
    'progress_callback': progress_callback,
    'revision': revision,
    'device': device,
    'dtype': dtype,
    'model_file_name': model_file_name,
    'session_options': session_options?.toJson(),
    'subfolder': subfolder,
    'use_external_data_format': use_external_data_format,
  };

  @override
  String toString() => jsonEncode(toJson());
}

/// Determines whether the given string is a valid URL.
/// @param {string|URL} string The string to test for validity as an URL.
/// @param {string[]} [protocols=null] A list of valid protocols. If specified, the protocol must be in this list.
/// Don't include ":" like in JavaScript as "http:" is "http" in dart.
/// @param {string[]} [validHosts=null] A list of valid hostnames. If specified, the URL's hostname must be in this list.
/// @returns {boolean} True if the string is a valid URL, false otherwise.
bool isValidUrl(String string, [List<String>? protocols, List<String>? validHosts]) {
  Uri url;
  try {
    url = Uri.parse(string);
  } catch (_) {
    return false;
  }
  if (protocols != null && !protocols.contains(url.scheme)) {
    return false;
  }
  if (validHosts != null && !validHosts.contains(url.host)) {
    return false;
  }
  return true;
}

final REPO_ID_REGEX = RegExp(r'^(\b[\w\-.]+\b\/)?\b[\w\-.]{1,96}\b$');

/// Tests whether a string is a valid Hugging Face model ID or not.
/// Adapted from https://github.com/huggingface/huggingface_hub/blob/6378820ebb03f071988a96c7f3268f5bdf8f9449/src/huggingface_hub/utils/_validators.py#L119-L170
///
/// @param {string} string The string to test
/// @returns {boolean} True if the string is a valid model ID, false otherwise.
bool isValidHfModelId(string) {
  if (!REPO_ID_REGEX.hasMatch(string)) return false;
  if (string.includes("..") || string.includes("--")) return false;
  if (string.endsWith(".git") || string.endsWith(".ipynb")) return false;
  return true;
}

/// Helper function to get a file, using either the Fetch API or FileSystem API.
///
/// @param {URL|string} urlOrPath The URL/path of the file to get.
/// @returns {Promise<FileResponse|Response>} A promise that resolves to a FileResponse object (if the file is retrieved using the FileSystem API), or a Response object (if the file is retrieved using the Fetch API).
Future<Uint8List> getFile(dynamic urlOrPath) async {
  if (urlOrPath is! String && urlOrPath is! Uri) {
    throw ArgumentError('Unsupported input type. Must be a String or Uri');
  }

  final String urlOrPathString = urlOrPath is Uri ? urlOrPath.toString() : urlOrPath;

  if (!isValidUrl(urlOrPathString, ['http', 'https', 'blob'])) {
    return File(urlOrPathString).readAsBytes();
  } else {
    final IS_CI = bool.tryParse(Platform.environment['TESTING_REMOTELY'] ?? '', caseSensitive: false) ?? false;
    final version = env.version;

    final Map<String, dynamic> headers = {};
    headers['User-Agent'] = 'transformers_dart/$version; is_ci/$IS_CI;';

    // Check whether we are making a request to the Hugging Face Hub.
    final isHFURL = isValidUrl(urlOrPath, ['http', 'https'], ['huggingface.co', 'hf.co']);
    if (isHFURL) {
      // If an access token is present in the environment variables,
      // we add it to the request headers.
      // NOTE: We keep `HF_ACCESS_TOKEN` for backwards compatibility (as a fallback).
      final token = Platform.environment['HF_TOKEN'] ?? Platform.environment['HF_ACCESS_TOKEN'];
      if (token != null) {
        headers['Authorization'] = 'Bearer $token';
      }
    }

    final response = await dio.get<List<int>>(
      urlOrPathString,
      options: Options(
        headers: headers,
        responseType: ResponseType.bytes,
      ),
    );

    if (response.statusCode == 200 && response.data != null) {
      return Uint8List.fromList(response.data!);
    } else {
      throw StateError('Unable to read image from "$urlOrPathString" (${response.statusCode} ${response.statusMessage})');
    }
  }
}

/// Retrieves a file from either a remote URL using the Fetch API or from the local file system using the FileSystem API.
/// If the filesystem is available and `env.useCache = true`, the file will be downloaded and cached.
///
/// @param {string} path_or_repo_id This can be either:
/// - a string, the *model id* of a model repo on huggingface.co.
/// - a path to a *directory* potentially containing the file.
/// @param {string} filename The name of the file to locate in `path_or_repo`.
/// @param {boolean} [fatal=true] Whether to throw an error if the file is not found.
/// @param {PretrainedOptions} [options] An object containing optional parameters.
/// @param {boolean} [return_path=false] Whether to return the path of the file instead of the file content.
///
/// @throws Will throw an error if the file is not found and `fatal` is true.
/// @returns {Promise<string|Uint8Array>} A Promise that resolves with the file content as a Uint8Array if `return_path` is false, or the file path as a string if `return_path` is true.
Future<String> getModelFile(String path_or_repo_id, String filename, [bool fatal = true, PretrainedOptions? options, bool return_path = false]) async {
  // print(path_or_repo_id);
  // print(filename);


  final String filePath = await hfHubDownload(
    repoId: path_or_repo_id,
    filename: filename,
  );
  if (return_path) return filePath;

  return await File(filePath).readAsString();


  // String? file_path;
  // String? user_home;
  // String? dir;
  //
  // final bool isLocalPath = await File(path_or_repo_id).exists();
  //
  // if (isLocalPath) {
  //   dir = path_or_repo_id;
  // } else {
  //   final String repoId = path_or_repo_id.replaceAll('/', '--');
  //
  //   // On web, there is no concept of a home directory
  //   if (kIsWeb) {
  //     return null;
  //   }
  //
  //   // Get the home directory from environment variables.
  //   // This is the most reliable way for desktop platforms.
  //   final Map<String, String> envVars = Platform.environment;
  //   if (Platform.isMacOS || Platform.isLinux) {
  //     user_home = envVars['HOME'];
  //   } else if (Platform.isWindows) {
  //     user_home = envVars['USERPROFILE'];
  //   } else {
  //     return null;
  //   }
  //
  //   if (user_home == null) return null;
  //
  //   final String hf_cache = path.join(user_home, '.cache', 'huggingface', 'hub');
  //   final String model_cache = path.join(hf_cache, 'models--$repoId');
  //   final String ref_path = path.join(model_cache, 'refs', 'main');
  //   final String snapshot_hash = await File(ref_path).readAsString();
  //   dir = path.join(model_cache, 'snapshots', snapshot_hash);
  // }
  //
  // file_path = path.join(dir, filename);
  // final file = File(file_path);
  //
  // return await file.readAsString();




  // if (!env.allowLocalModels) {
  //   // User has disabled local models, so we just make sure other settings are correct.
  //
  //   if (options.local_files_only) {
  //     throw Exception('Invalid configuration detected: local models are disabled (`env.allowLocalModels=false`) but you have requested to only use local models (`local_files_only=true`).')
  //   } else if (!env.allowRemoteModels) {
  //     throw Exception('Invalid configuration detected: both local and remote models are disabled. Fix by setting `env.allowLocalModels` or `env.allowRemoteModels` to `true`.')
  //   }
  // }
  //
  // // Initiate file retrieval
  // dispatchCallback(options.progress_callback, InitiateProgressInfo(path_or_repo_id, filename));
  //
  // // First, check if the a caching backend is available
  // // If no caching mechanism available, will download the file every time
  // var cache;
  // if (!cache && env.useCustomCache) {
  //   // Allow the user to specify a custom cache system.
  //   if (!env.customCache) {
  //     throw Exception('`env.useCustomCache=true`, but `env.customCache` is not defined.');
  //   }
  //
  //   // Check that the required methods are defined:
  //   if (!env.customCache.match || !env.customCache.put) {
  //     throw Exception('`env.customCache` must be an object which implements the `match` and `put` functions of the Web Cache API. For more information, see https://developer.mozilla.org/en-US/docs/Web/API/Cache');
  //   }
  //   cache = env.customCache;
  // }
  //
  // if (!cache && env.useBrowserCache) {
  //   if (typeof caches === 'undefined') {
  //     throw Exception('Browser cache is not available in this environment.');
  //   }
  //
  //   try {
  //     // In some cases, the browser cache may be visible, but not accessible due to security restrictions.
  //     // For example, when running an application in an iframe, if a user attempts to load the page in
  //     // incognito mode, the following error is thrown: `DOMException: Failed to execute 'open' on 'CacheStorage':
  //     // An attempt was made to break through the security policy of the user agent.`
  //     // So, instead of crashing, we just ignore the error and continue without using the cache.
  //     cache = await caches.open('transformers-cache');
  //   } catch (e) {
  //     console.warn('An error occurred while opening the browser cache:', e);
  //   }
  // }
  //
  // if (!cache && env.useFSCache) {
  //   if (!apis.IS_FS_AVAILABLE) {
  //     throw Exception('File System Cache is not available in this environment.');
  //   }
  //
  //   // If `cache_dir` is not specified, use the default cache directory
  //   cache = new FileCache(options.cache_dir ?? env.cacheDir);
  // }
  //
  // final revision = options.revision ?? 'main';
  // final requestURL = pathJoin(path_or_repo_id, filename);
  //
  // final validModelId = isValidHfModelId(path_or_repo_id);
  // final localPath = validModelId ? pathJoin(env.localModelPath, requestURL) : requestURL;
  // final remoteURL = pathJoin(
  //   env.remoteHost,
  //   env.remotePathTemplate
  //     .replaceAll('{model}', path_or_repo_id)
  //     .replaceAll('{revision}', encodeURIComponent(revision)),
  //   filename
  // );
  //
  // /** @type {string} */
  // String? cacheKey;
  // final proposedCacheKey = cache instanceof FileCache
  // // Choose cache key for filesystem cache
  // // When using the main revision (default), we use the request URL as the cache key.
  // // If a specific revision is requested, we account for this in the cache key.
  // ? revision === 'main' ? requestURL : pathJoin(path_or_repo_id, revision, filename)
  //     : remoteURL;
  //
  // // Whether to cache the final response in the end.
  // bool toCacheResponse = false;
  //
  // /** @type {Response|FileResponse|undefined} */
  // let response;
  //
  // if (cache) {
  // // A caching system is available, so we try to get the file from it.
  // //  1. We first try to get from cache using the local path. In some environments (like deno),
  // //     non-URL cache keys are not allowed. In these cases, `response` will be undefined.
  // //  2. If no response is found, we try to get from cache using the remote URL or file system cache.
  // response = await tryCache(cache, localPath, proposedCacheKey);
  // }
  //
  // const cacheHit = response !== undefined;
  // if (response === undefined) {
  // // Caching not available, or file is not cached, so we perform the request
  //
  // if (env.allowLocalModels) {
  // // Accessing local models is enabled, so we try to get the file locally.
  // // If request is a valid HTTP URL, we skip the local file check. Otherwise, we try to get the file locally.
  // const isURL = isValidUrl(requestURL, ['http:', 'https:']);
  // if (!isURL) {
  // try {
  // response = await getFile(localPath);
  // cacheKey = localPath; // Update the cache key to be the local path
  // } catch (e) {
  // // Something went wrong while trying to get the file locally.
  // // NOTE: error handling is done in the next step (since `response` will be undefined)
  // console.warn(`Unable to load from local path "${localPath}": "${e}"`);
  // }
  // } else if (options.local_files_only) {
  // throw new Error(`\`local_files_only=true\`, but attempted to load a remote file from: ${requestURL}.`);
  // } else if (!env.allowRemoteModels) {
  // throw new Error(`\`env.allowRemoteModels=false\`, but attempted to load a remote file from: ${requestURL}.`);
  // }
  // }
  //
  // if (response === undefined || response.status === 404) {
  // // File not found locally. This means either:
  // // - The user has disabled local file access (`env.allowLocalModels=false`)
  // // - the path is a valid HTTP url (`response === undefined`)
  // // - the path is not a valid HTTP url and the file is not present on the file system or local server (`response.status === 404`)
  //
  // if (options.local_files_only || !env.allowRemoteModels) {
  // // User requested local files only, but the file is not found locally.
  // if (fatal) {
  // throw Error(`\`local_files_only=true\` or \`env.allowRemoteModels=false\` and file was not found locally at "${localPath}".`);
  // } else {
  // // File not found, but this file is optional.
  // // TODO in future, cache the response?
  // return null;
  // }
  // }
  // if (!validModelId) {
  // // Before making any requests to the remote server, we check if the model ID is valid.
  // // This prevents unnecessary network requests for invalid model IDs.
  // throw Error(`Local file missing at "${localPath}" and download aborted due to invalid model ID "${path_or_repo_id}".`);
  // }
  //
  // // File not found locally, so we try to download it from the remote server
  // response = await getFile(remoteURL);
  //
  // if (response.status !== 200) {
  // return handleError(response.status, remoteURL, fatal);
  // }
  //
  // // Success! We use the proposed cache key from earlier
  // cacheKey = proposedCacheKey;
  // }
  //
  // // Only cache the response if:
  // toCacheResponse =
  // cache                              // 1. A caching system is available
  // && typeof Response !== 'undefined' // 2. `Response` is defined (i.e., we are in a browser-like environment)
  // && response instanceof Response    // 3. result is a `Response` object (i.e., not a `FileResponse`)
  // && response.status === 200         // 4. request was successful (status code 200)
  // }
  //
  // // Start downloading
  // dispatchCallback(options.progress_callback, {
  // status: 'download',
  // name: path_or_repo_id,
  // file: filename
  // })
  //
  // let result;
  // if (!(apis.IS_NODE_ENV && return_path)) {
  // /** @type {Uint8Array} */
  // let buffer;
  //
  // if (!options.progress_callback) {
  // // If no progress callback is specified, we can use the `.arrayBuffer()`
  // // method to read the response.
  // buffer = new Uint8Array(await response.arrayBuffer());
  //
  // } else if (
  // cacheHit // The item is being read from the cache
  // &&
  // typeof navigator !== 'undefined' && /firefox/i.test(navigator.userAgent) // We are in Firefox
  // ) {
  // // Due to bug in Firefox, we cannot display progress when loading from cache.
  // // Fortunately, since this should be instantaneous, this should not impact users too much.
  // buffer = new Uint8Array(await response.arrayBuffer());
  //
  // // For completeness, we still fire the final progress callback
  // dispatchCallback(options.progress_callback, {
  // status: 'progress',
  // name: path_or_repo_id,
  // file: filename,
  // progress: 100,
  // loaded: buffer.length,
  // total: buffer.length,
  // })
  // } else {
  // buffer = await readResponse(response, data => {
  // dispatchCallback(options.progress_callback, {
  // status: 'progress',
  // name: path_or_repo_id,
  // file: filename,
  // ...data,
  // })
  // })
  // }
  // result = buffer;
  // }
  //
  // if (
  // // Only cache web responses
  // // i.e., do not cache FileResponses (prevents duplication)
  // toCacheResponse && cacheKey
  // &&
  // // Check again whether request is in cache. If not, we add the response to the cache
  // (await cache.match(cacheKey) === undefined)
  // ) {
  // if (!result) {
  // // We haven't yet read the response body, so we need to do so now.
  // await cache.put(cacheKey, /** @type {Response} */(response), options.progress_callback);
  // } else {
  // // NOTE: We use `new Response(buffer, ...)` instead of `response.clone()` to handle LFS files
  // await cache.put(cacheKey, new Response(result, {
  // headers: response.headers
  // }))
  //     .catch(err => {
  // // Do not crash if unable to add to cache (e.g., QuotaExceededError).
  // // Rather, log a warning and proceed with execution.
  // console.warn(`Unable to add response to browser cache: ${err}.`);
  // });
  // }
  // }
  // dispatchCallback(options.progress_callback, {
  // status: 'done',
  // name: path_or_repo_id,
  // file: filename
  // });
  //
  // if (result) {
  // if (!apis.IS_NODE_ENV && return_path) {
  // throw new Error("Cannot return path in a browser environment.")
  // }
  // return result;
  // }
  // if (response instanceof FileResponse) {
  // return response.filePath;
  // }
  //
  // // Otherwise, return the cached response (most likely a `FileResponse`).
  // // NOTE: A custom cache may return a Response, or a string (file path)
  // const cachedResponse = await cache?.match(cacheKey);
  // if (cachedResponse instanceof FileResponse) {
  // return cachedResponse.filePath;
  // } else if (cachedResponse instanceof Response) {
  // return new Uint8Array(await cachedResponse.arrayBuffer());
  // } else if (typeof cachedResponse === 'string') {
  // return cachedResponse;
  // }
  //
  // throw Exception('Unable to get model file path or buffer.');
}

/// Fetches a JSON file from a given path and file name.
///
/// @param {string} modelPath The path to the directory containing the file.
/// @param {string} fileName The name of the file to fetch.
/// @param {boolean} [fatal=true] Whether to throw an error if the file is not found.
/// @param {PretrainedOptions} [options] An object containing optional parameters.
/// @returns {Promise<Object>} The JSON data parsed into a JavaScript object.
/// @throws Will throw an error if the file is not found and `fatal` is true.
Future<Map<String, dynamic>> getModelJSON(String modelPath, String fileName, [bool fatal = true, PretrainedOptions? options]) async {
  final buffer = await getModelFile(modelPath, fileName, fatal, options, false);
  if (buffer == null) return {};

  // final decoder = new TextDecoder('utf-8');
  // const jsonData = decoder.decode(/** @type {Uint8Array} */(buffer));

  return jsonDecode(buffer);
}
