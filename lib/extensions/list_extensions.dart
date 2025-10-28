extension ListExtensions on List {
  List flat([int depth = 1]) => depth <= 0
      ? this
      : expand((x) => x).toList().flat(depth - 1);
}
