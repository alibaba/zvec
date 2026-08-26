#include <iostream>
#include <zvec/db/status.h>

int main() {
  const char *message = zvec::GetDefaultMessage(zvec::StatusCode::OK);
  if (!message) {
    std::cerr << "cpp_smoke: GetDefaultMessage() returned null" << std::endl;
    return 1;
  }
  std::cout << "cpp_smoke: StatusCode::OK -> " << message << std::endl;
  std::cout << "cpp_smoke: OK" << std::endl;
  return 0;
}
