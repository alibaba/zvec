#include <iostream>
#include <string>
#include <zvec/ailego/utility/string_helper.h>

using namespace zvec;

int main() {
  std::string a{"hello world"};

  std::cout << ailego::StringHelper::StartsWith(a, "hello") << std::endl;
}