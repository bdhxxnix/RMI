#include <cstddef>
#include <cstdint>
namespace rmi {
bool load(char const* dataPath);
void cleanup();
const size_t RMI_SIZE = 2416;
const uint64_t BUILD_TIME_NS = 10526876207;
const char NAME[] = "rmi";
uint64_t lookup(uint64_t key, size_t* err);
}
