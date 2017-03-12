//STL
#include <type_traits>

template<int Begin, int End, int Val, class Enable = void>
struct ctrange { };

template<int Begin, int End, int Val>
struct ctrange<Begin, End, Val,
  typename std::enable_if<Val >= Begin && Val < End>::type> {
  using enabled = void;
};

