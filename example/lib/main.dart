import 'dart:async';
import 'dart:isolate';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:transformers/transformers.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Transformers.ensureInitialized(throwOnFail: true);
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class SimilaritySearchIsolate {
  /// Spawn a new [Isolate] that handles similarity search.
  static Future<SimilaritySearchIsolate> spawnIsolate() async {
    final receivePort = ReceivePort();
    final Stream receivePortStream = receivePort.asBroadcastStream();
    late SendPort sendPort;
    final cls = SimilaritySearchIsolate._();
    final Completer<void> isolateReady = Completer();

    final StreamSubscription subscription = receivePortStream.listen((message) {
      if (message is SendPort) {
        sendPort = message;
        isolateReady.complete();
      }
    });

    final RootIsolateToken rootIsolateToken = ServicesBinding.rootIsolateToken!;
    final isolate = await Isolate.spawn(SimilaritySearchIsolate._create, (receivePort.sendPort, rootIsolateToken));

    await isolateReady.future;
    await subscription.cancel();
    await cls._initClient(isolate, receivePort, receivePortStream, sendPort);
    return cls;
  }

  /// The entrypoint for [Isolate.spawn].
  static Future<void> _create((SendPort sendPort, RootIsolateToken rootIsolateToken) args) async {
    final isolate = SimilaritySearchIsolate._();
    await isolate._initIsolate(args.$1, args.$2);
  }

  // --- Below is shared in both the client and isolate ---
  late final SendPort _sendPort;
  ReceivePort _receivePort = ReceivePort();
  late Stream _receivePortStream;

  SimilaritySearchIsolate._();

  // --- Below is for the client ---

  late final Isolate _isolate;

  Future<void> _initClient(Isolate isolate, ReceivePort receivePort, Stream receivePortStream, SendPort sendPort) async {
    _isolate = isolate;
    _receivePort = receivePort;
    _receivePortStream = receivePortStream;
    _sendPort = sendPort;
  }

  Future<List<({String text, double likelyhood})>> similaritySearch(List<String> texts, String query) async {
    final Completer<void> messageReceived = Completer();
    late List<double> results;

    final StreamSubscription subscription = _receivePortStream.listen((message) {
      if (message is List<double>) {
        results = message;
        messageReceived.complete();
      }
    });

    _sendPort.send({'query': query, 'texts': texts});

    await messageReceived.future;
    await subscription.cancel();

    return results.indexed
        .map((e) => (text: texts[e.$1], likelyhood: e.$2))
        .toList()
      ..sort((a, b) => b.likelyhood.compareTo(a.likelyhood));
  }

  void dispose() {
    _isolate.kill();
    _receivePort.close();
  }

  // --- Below is for the isolate ---

  late final Pipeline _extractor;

  Future<void> _initIsolate(SendPort sendPort, RootIsolateToken rootIsolateToken) async {
    _sendPort = sendPort;
    _sendPort.send(_receivePort.sendPort);

    BackgroundIsolateBinaryMessenger.ensureInitialized(rootIsolateToken);
    await Transformers.ensureInitialized();

    _extractor = await pipeline(
      PipelineType.featureExtraction,
      'onnx-community/Qwen3-Embedding-0.6B-ONNX',
      PretrainedModelOptions(
        dtype: DataType.fp32,
      ),
    );

    _receivePort.listen((message) async {
      final query = message['query'] as String;
      final texts = message['texts'] as List<String>;

      String getDetailedInstruct(String taskDescription, String query) {
        return 'Instruct: $taskDescription\nQuery:$query';
      }

      // Each query must come with a one-sentence instruction that describes the task
      const instruct = "Given a web search query, retrieve relevant passages that are close to the query";
      final queries = [
        getDetailedInstruct(instruct, query),
      ];

      // No need to add instruction for retrieval documents
      final inputTexts = [...queries, ...texts];

      // Extract embeddings for queries and documents
      final Tensor output = await _extractor(inputTexts, FeatureExtractionPipelineOptions(
        pooling: "last_token",
        normalize: true,
      ));
      final scores = await matmul(
        await output.slice([[0, queries.length]]), // Query embeddings
        await (await output.slice([[queries.length, null]])).transpose([1, 0]), // Document embeddings
      );

      _sendPort.send(scores.data.cast<double>());
    });
  }
}

class _MyAppState extends State<MyApp> {
  static const List<String> _defaultTexts = [
    'The quick brown fox jumps over the lazy dog.',
    'A stitch in time saves nine.',
    'Actions speak louder than words.',
    'All that glitters is not gold.',
    'The early bird catches the worm.',
    'Honesty is the best policy.',
    'Laughter is the best medicine.',
    'The pen is mightier than the sword.',
    'There is no time like the present.',
    'Practice makes perfect.',
    'Where there is a will, there is a way.',
    'You can lead a horse to water, but you can\'t make it drink.',
    'The squeaky wheel gets the grease.',
    'Two heads are better than one.',
    'When in Rome, do as the Romans do.',
    'A picture is worth a thousand words.',
    'Beauty is in the eye of the beholder.',
    'Don\'t count your chickens before they hatch.',
    'Every cloud has a silver lining.',
    'Fortune favors the bold.',
    'Good things come to those who wait.',
    'Hope for the best, but prepare for the worst.',
    'If it ain\'t broke, don\'t fix it.',
    'Knowledge is power.',
    'Look before you leap.'
  ];

  late final SimilaritySearchIsolate _searchIsolate;
  bool _ready = false;
  final TextEditingController _searchController = TextEditingController();
  final TextEditingController _addTextController = TextEditingController();
  final TextEditingController _resultsController = TextEditingController();

  late List<String> _texts = _defaultTexts.toList();

  @override
  void initState() {
    super.initState();
    _spawnIsolate();
  }

  @override
  void dispose() {
    _searchIsolate.dispose();
    _searchController.dispose();
    _addTextController.dispose();
    _resultsController.dispose();
    super.dispose();
  }

  void _spawnIsolate() async {
    _searchIsolate = await SimilaritySearchIsolate.spawnIsolate();
    setState(() {
      _ready = true;
    });
  }

  void _runSimilaritySearch() async {
    final query = _searchController.text;
    if (query.isEmpty) return;
    if (!_ready) {
      setState(() {
        _resultsController.text = 'Process is starting up...';
      });
      return;
    }

    final results = await _searchIsolate.similaritySearch(_texts, query);

    setState(() {
      _resultsController.text = results.join('\n');
    });
  }

  void _addText() {
    if (_addTextController.text.isNotEmpty) {
      setState(() {
        _texts.add(_addTextController.text);
        _addTextController.clear();
      });
      _runSimilaritySearch();
    }
  }

  void _removeLastText() {
    if (_texts.isNotEmpty) {
      setState(() {
        _texts.removeLast();
      });
      _runSimilaritySearch();
    }
  }

  void _resetTexts() {
    setState(() {
      _texts = _defaultTexts.toList();
    });
    _runSimilaritySearch();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Transformers Similarity Search'),
        ),
        body: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  Expanded(
                    child: TextField(
                      controller: _searchController,
                      decoration: const InputDecoration(
                        labelText: 'Enter text for similarity search',
                      ),
                    ),
                  ),
                  const SizedBox(width: 8.0),
                  ElevatedButton(
                    onPressed: _runSimilaritySearch,
                    child: const Text('Search'),
                  ),
                ],
              ),
              const SizedBox(height: 16.0),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  Expanded(
                    child: TextField(
                      controller: _addTextController,
                      decoration: const InputDecoration(
                        labelText: 'Enter text to add',
                      ),
                    ),
                  ),
                  const SizedBox(width: 8.0),
                  ElevatedButton(
                    onPressed: _addText,
                    child: const Text('Add'),
                  ),
                  const SizedBox(width: 8.0),
                  ElevatedButton(
                    onPressed: _removeLastText,
                    child: const Text('Remove Last'),
                  ),
                  const SizedBox(width: 8.0),
                  ElevatedButton(
                    onPressed: _resetTexts,
                    child: const Text('Reset'),
                  ),
                ],
              ),
              const SizedBox(height: 16.0),
              const Text(
                'Items for Similarity Search:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              Expanded(
                child: ListView.builder(
                  itemCount: _texts.length,
                  itemBuilder: (context, index) {
                    return Card(
                      child: ListTile(
                        title: Text(_texts[index]),
                      ),
                    );
                  },
                ),
              ),
              const SizedBox(height: 16.0),
              const Text(
                'Similarity Search Results:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              Expanded(
                child: TextField(
                  controller: _resultsController,
                  readOnly: true,
                  maxLines: null,
                  decoration: const InputDecoration(
                    border: OutlineInputBorder(),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}