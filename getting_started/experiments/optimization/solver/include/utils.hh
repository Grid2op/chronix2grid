void print_cpp_version() {
  printf("Compiled using ");
  if (__cplusplus == 202002L) printf("C++20\n");
  else if (__cplusplus == 201703L) printf("C++17\n");
  else if (__cplusplus == 201402L) printf("C++14\n");
  else if (__cplusplus == 201103L) printf("C++11\n");
  else if (__cplusplus == 199711L) printf("C++98\n");
  else printf("pre-standard C++\n");
}