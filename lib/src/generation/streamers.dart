import 'dart:async';
import 'dart:io';

import 'package:transformers/src/tokenizers.dart';
import 'package:transformers/src/utils/core.dart';

class BaseStreamer {
  /// Function that is called by `.generate()` to push new tokens
  /// @param {bigint[][]} value
  void put(List<List<int>> value) {
    throw UnimplementedError('Not implemented');
  }

  /// Function that is called by `.generate()` to signal the end of generation
  void end() {
    throw UnimplementedError('Not implemented');
  }
}

void stdout_write(Object? x) => stdout.write(x);

/// Simple text streamer that prints the token(s) to stdout as soon as entire words are formed.
class TextStreamer extends BaseStreamer {
  PreTrainedTokenizer tokenizer;

  /// Whether to skip the prompt tokens
  bool skip_prompt;

  /// Function to call when a piece of text is ready to display
  void Function(String)? callback_function;

  /// Function to call when a new token is generated
  void Function(List<int>)? token_callback_function;

  /// Additional keyword arguments to pass to the tokenizer's decode method
  Map<String, dynamic> decode_kwargs;

  List<int> token_cache = [];

  int print_len = 0;

  bool next_tokens_are_prompt = true;

  /// @param {import('../tokenizers.js').PreTrainedTokenizer} tokenizer
  /// @param {Object} options
  /// @param {boolean} [options.skip_prompt=false] Whether to skip the prompt tokens
  /// @param {boolean} [options.skip_special_tokens=true] Whether to skip special tokens when decoding
  /// @param {function(string): void} [options.callback_function=null] Function to call when a piece of text is ready to display
  /// @param {function(bigint[]): void} [options.token_callback_function=null] Function to call when a new token is generated
  /// @param {Object} [options.decode_kwargs={}] Additional keyword arguments to pass to the tokenizer's decode method
  TextStreamer(this.tokenizer, [Map<String, dynamic> args = const {}])
    : skip_prompt = args.remove('skip_prompt') ?? false,
      callback_function = args.remove('callback_function') ?? stdout_write,
      token_callback_function = args.remove('token_callback_function'),
      decode_kwargs = {
        'skip_special_tokens': args['skip_special_tokens'],
        ...args['decode_kwargs'] ?? {},
        ...args,
      };

  /// Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
  /// @param {bigint[][]} value
  @override
  void put(List<List<int>> value) {
    if (value.length > 1) {
      throw ArgumentError('TextStreamer only supports batch size of 1');
    }

    final is_prompt = next_tokens_are_prompt;
    if (is_prompt) {
      next_tokens_are_prompt = false;
      if (skip_prompt) return;
    }

    final List<int> tokens = value[0];
    token_callback_function?.call(tokens);

    // Add the new token to the cache and decodes the entire thing.
    token_cache = mergeArrays([token_cache, tokens]);
    final text = tokenizer.decode(
      token_cache,
      clean_up_tokenization_spaces: decode_kwargs['clean_up_tokenization_spaces'],
      skip_special_tokens: decode_kwargs['skip_special_tokens'],
    );

    String printable_text;
    if (is_prompt || text.endsWith('\n')) {
      // After the symbol for a new line, we flush the cache.
      printable_text = text.substring(print_len);
      token_cache = [];
      print_len = 0;
    } else if (text.isNotEmpty && is_chinese_char(text.codeUnitAt(text.length - 1))) {
      // If the last token is a CJK character, we print the characters.
      printable_text = text.substring(print_len);
      print_len += printable_text.length;
    } else {
      // Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
      // which may change with the subsequent token -- there are probably smarter ways to do this!)
      printable_text = text.substring(print_len, text.lastIndexOf(' ') + 1);
      print_len += printable_text.length;
    }

    on_finalized_text(printable_text, false);
  }

  /// Flushes any remaining cache and prints a newline to stdout.
  @override
  void end() {
    String printable_text;
    if (token_cache.isNotEmpty) {
      final text = tokenizer.decode(
        token_cache,
        clean_up_tokenization_spaces: decode_kwargs['clean_up_tokenization_spaces'],
        skip_special_tokens: decode_kwargs['skip_special_tokens'],
      );
      printable_text = text.substring(print_len);
      token_cache = [];
      print_len = 0;
    } else {
      printable_text = '';
    }
    next_tokens_are_prompt = true;
    on_finalized_text(printable_text, true);
  }

  /// Prints the new text to stdout. If the stream is ending, also prints a newline.
  /// @param {string} text
  /// @param {boolean} stream_end
  void on_finalized_text(String text, bool stream_end) {
    if (text.isNotEmpty) {
      callback_function?.call(text);
    }
    if (stream_end && callback_function == stdout_write) {
      callback_function?.call('\n');
    }
  }
}
