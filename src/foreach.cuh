#pragma once
#include <tuple>

/// Helper functions to navigate through a variadic number of fields at compile-time.

// Apply lambda to each element of a tuple (a compile-time ranged-based for loop).
template <std::size_t index = 0, typename Function, typename... T>
inline typename std::enable_if<index == sizeof...(T), void>::type
for_each(std::tuple<T...> &, Function) { }
template <std::size_t index = 0, typename Function, typename... T>
inline typename std::enable_if<index < sizeof...(T), void>::type
for_each(std::tuple<T...>& t, Function f)
{
    f(index, std::get<index>(t));
    for_each<index + 1, Function, T...>(t, f);
}

// Apply lambda function as a more conventional "for loop" (but at compile-time).
template <std::size_t... I, typename Function>
inline void for_each(std::index_sequence<I...>, Function func)
{
    (func(std::integral_constant<std::size_t, I>{}), ...);
}
