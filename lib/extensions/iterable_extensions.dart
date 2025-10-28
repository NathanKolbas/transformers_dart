extension IterableExtensions on Iterable {
  Iterable flat([int depth = 1]) => depth <= 0
      ? this
      : expand((x) => x).toList().flat(depth - 1);
}
