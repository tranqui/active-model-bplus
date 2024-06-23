#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

TEST_CASE("MyTestSuite/Test1", "Test1")
{
    REQUIRE(2 + 2 == 4);
}