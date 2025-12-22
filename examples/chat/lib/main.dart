import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:transformers/transformers.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Transformers.ensureInitialized(throwOnFail: true);

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Transformers Chat',
      theme: ThemeData(
        colorScheme: .fromSeed(seedColor: Colors.deepPurple),
      ),
      darkTheme: ThemeData.dark().copyWith(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple, brightness: Brightness.dark),
      ),
      themeMode: .system,
      debugShowCheckedModeBanner: false,
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final TextEditingController _textController = TextEditingController();

  String _modelId = 'onnx-community/granite-4.0-350m-ONNX';
  final PipelineType task = PipelineType.textGeneration;
  Pipeline? generator;

  bool showModelLoader = false;
  Object? loadingModelError;
  String systemMessage = 'You are a helpful assistant.';
  List<Message> _chats = [];

  void setSystemMessage(String message) {
    setState(() {
      systemMessage = message;
      _chats.removeAt(0);
      _chats.insert(0, Message(role: 'system', content: systemMessage));
    });
  }

  Future<void> setModelId(String id) async {
    setState(() {
      _modelId = id;
    });
    await setupModel();
  }

  void openModelIdDialog() => showDialog(
    context: context,
    builder: (context) => SetModelIdDialog(
      modelId: _modelId,
      setModelId: setModelId,
    ),
  );

  void resetChat() {
    setState(() {
      _chats = [
        Message(role: 'system', content: systemMessage),
      ];
    });
  }

  void sendMessage(String message) {
    message = message.trim();
    if (message.trim().isEmpty) return;

    _textController.clear();

    setState(() {
      _chats.add(Message(role: 'user', content: message));
    });

    generateResponse();
  }

  Future<void> setupModel() async {
    setState(() {
      showModelLoader = true;
      loadingModelError = null;
    });
    try {
      generator = await pipeline(task, _modelId, PretrainedModelOptions());
      setState(() {
        showModelLoader = false;
      });
    } catch (e) {
      setState(() {
        loadingModelError = e;
      });
    }
  }

  Future<void> generateResponse() async {
    if (this.generator == null) {
      await setupModel();
    }
    final generator = this.generator!;

    _chats.add(Message(role: 'assistant', content: ''));
    void streamingResponse(String text) => setState(() {
      final content = _chats.last.content + text;
      _chats.removeLast();
      _chats.add(Message(role: 'assistant', content: content));
    });

    await generator(_chats, TextGenerationConfig(
      max_new_tokens: 512,
      do_sample: false,
      streamer: TextStreamer(generator.tokenizer!, {
        'skip_prompt': true,
        'skip_special_tokens': true,
        'callback_function': streamingResponse,
      }),
    ));
  }

  @override
  void initState() {
    resetChat();
    super.initState();
    setupModel();
  }

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          Column(
            children: [
              Column(
                children: [
                  AppBar(
                    backgroundColor: Colors.transparent,
                    centerTitle: true,
                    title: Text('Transformers Chat'),
                    actions: [
                      IconButton.filledTonal(
                        onPressed: openModelIdDialog,
                        icon: Icon(Icons.smart_toy_outlined),
                      ),
                      const SizedBox(width: 8,),
                      IconButton.filledTonal(
                        onPressed: () => showDialog(
                          context: context,
                          builder: (context) => SetSystemMessageDialog(
                            systemMessage: systemMessage,
                            setSystemMessage: setSystemMessage,
                          ),
                        ),
                        icon: Icon(Icons.message_outlined),
                      ),
                      const SizedBox(width: 8,),
                      IconButton.filledTonal(
                        onPressed: resetChat,
                        icon: Icon(Icons.refresh_rounded),
                      ),
                    ],
                    actionsPadding: EdgeInsets.only(right: 12),
                  ),
                  Container(
                    color: Theme.of(context).colorScheme.primaryContainer,
                    padding: EdgeInsets.all(4),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        SelectableText.rich(
                          TextSpan(
                            text: 'Current model is: ',
                            children: [
                              TextSpan(
                                text: _modelId,
                                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                  fontStyle: .italic,
                                ),
                              ),
                            ],
                          ),
                          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            fontWeight: .bold,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              Expanded(
                child: ListView.separated(
                  separatorBuilder: (_, _) => const SizedBox(height: 12,),
                  itemCount: _chats.length,
                  reverse: true,
                  itemBuilder: (context, i) {
                    final message = _chats[_chats.length - 1 - i];

                    if ((message.content as String).trim().isEmpty) {
                      return SizedBox.shrink();
                    }

                    return switch (message.role) {
                      'system' => SystemMessage(message: message),
                      'user' => UserMessage(message: message),
                      'assistant' => AssistantMessage(message: message),
                      _ => const Placeholder(),
                    };
                  },
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Card(
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadiusGeometry.circular(32),
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        child: Padding(
                          padding: const EdgeInsets.only(left: 16.0, right: 16.0),
                          child: TextField(
                            focusNode: FocusNode(
                              onKeyEvent: (FocusNode node, KeyEvent evt) {
                                if (!HardwareKeyboard.instance.isShiftPressed &&
                                    evt.logicalKey.keyLabel == 'Enter') {
                                  if (evt is KeyDownEvent) {
                                    sendMessage(_textController.text);
                                  }
                                  return KeyEventResult.handled;
                                } else {
                                  return KeyEventResult.ignored;
                                }
                              },
                            ),
                            controller: _textController,
                            onSubmitted: sendMessage,
                            minLines: 1,
                            maxLines: null,
                            keyboardType: .multiline,
                            textCapitalization: .sentences,
                            decoration: InputDecoration(
                              hintText: 'Send a message',
                              border: InputBorder.none,
                            ),
                          ),
                        ),
                      ),
                      IconButton(
                        onPressed: () => sendMessage(_textController.text),
                        icon: Icon(
                          Icons.send_rounded,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
          if (showModelLoader) Center(
            child: ClipRect(
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 10.0, sigmaY: 10.0),
                child: Center(
                  child: SingleChildScrollView(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: loadingModelError == null ? [
                        SelectableText.rich(
                          TextSpan(
                            text: 'Loading model\n',
                            children: [
                              TextSpan(
                                text: _modelId,
                                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                  fontStyle: .italic,
                                ),
                              ),
                            ],
                          ),
                          textAlign: TextAlign.center,
                          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            fontWeight: .bold,
                          ),
                        ),
                        const SizedBox(height: 32,),
                        CircularProgressIndicator(),
                      ] : [
                        SelectableText.rich(
                          TextSpan(
                            text: 'Error loading model\n',
                            children: [
                              TextSpan(
                                text: _modelId,
                                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                  fontStyle: .italic,
                                ),
                              ),
                              TextSpan(text: '\n\n'),
                              WidgetSpan(
                                child: SelectableText.rich(
                                  TextSpan(text: '$loadingModelError'),
                                  textAlign: TextAlign.start,
                                  style: Theme.of(context).textTheme.bodySmall,
                                ),
                              ),
                            ],
                          ),
                          textAlign: TextAlign.center,
                          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            fontWeight: .bold,
                          ),
                        ),
                        const SizedBox(height: 32,),
                        Icon(
                          Icons.error_outline_rounded,
                          color: Colors.red,
                          size: 64,
                        ),
                        const SizedBox(height: 32,),
                        FilledButton.tonal(
                          onPressed: openModelIdDialog,
                          child: const Text('Select a different model'),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class SetSystemMessageDialog extends StatefulWidget {
  final String systemMessage;
  final void Function(String) setSystemMessage;

  const SetSystemMessageDialog({super.key, required this.systemMessage, required this.setSystemMessage});

  @override
  State<SetSystemMessageDialog> createState() => _SetSystemMessageDialogState();
}

class _SetSystemMessageDialogState extends State<SetSystemMessageDialog> {
  late final controller = TextEditingController(text: widget.systemMessage);

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Set System Message'),
      content: TextField(
        controller: controller,
        decoration: const InputDecoration(
          hintText: 'Enter system message',
        ),
        onSubmitted: (value) {
          widget.setSystemMessage(value);
          Navigator.of(context).pop();
        },
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        TextButton(
          onPressed: () {
            widget.setSystemMessage(controller.text);
            Navigator.of(context).pop();
          },
          child: const Text('Set'),
        ),
      ],
    );
  }
}

class SetModelIdDialog extends StatefulWidget {
  final String modelId;
  final void Function(String) setModelId;

  const SetModelIdDialog({super.key, required this.modelId, required this.setModelId});

  @override
  State<SetModelIdDialog> createState() => _SetModelIdDialogState();
}

class _SetModelIdDialogState extends State<SetModelIdDialog> {
  late final TextEditingController _modelIdController = TextEditingController(text: widget.modelId);

  void submit() {
    if (_modelIdController.text.trim().isEmpty) return;

    widget.setModelId(_modelIdController.text);
    Navigator.of(context).pop();
  }

  @override
  void dispose() {
    _modelIdController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Huggingface Hub Model ID'),
      content: TextField(
        controller: _modelIdController,
        decoration: const InputDecoration(
          hintText: 'Enter model ID',
        ),
        onSubmitted: (value) => submit(),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        TextButton(
          onPressed: submit,
          child: const Text('Set'),
        ),
      ],
    );
  }
}


class SystemMessage extends StatelessWidget {
  final Message message;

  const SystemMessage({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        SelectableText.rich(
          TextSpan(
            text: 'System: ',
            children: [
              TextSpan(
                text: message.content,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  fontStyle: .italic,
                ),
              ),
            ],
          ),
          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
            fontWeight: .bold,
          ),
        ),
      ],
    );
  }
}

class UserMessage extends StatelessWidget {
  final Message message;

  const UserMessage({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(left: 32.0, right: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Card(
            color: Theme.of(context).colorScheme.primaryContainer,
            child: Padding(
              padding: const EdgeInsets.all(8.0),
              child: SelectableText(
                message.content,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class AssistantMessage extends StatelessWidget {
  final Message message;

  const AssistantMessage({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(left: 8, right: 32.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Card(
            color: Theme.of(context).colorScheme.tertiaryContainer,
            child: Padding(
              padding: const EdgeInsets.all(8.0),
              child: SelectableText(
                message.content,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
