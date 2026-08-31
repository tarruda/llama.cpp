#include "testing.h"

#include "mtmd-image.h"
#include "mtmd-internal.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// this test file contains:
// 1. test cases for mtmd helpers
// 2. test cases for internal mtmd components
// internal headers can be included here

struct test_registry {
    using fn_t = void (*)(testing &);

    struct entry {
        std::string name;
        fn_t fn;
    };

    static std::vector<entry> & all() {
        static std::vector<entry> entries;
        return entries;
    }

    test_registry(const char * name, fn_t fn) {
        all().push_back({ name, fn });
    }
};

#define MAKE_TEST(name)                                               \
    static void name(testing & t);                                    \
    static const test_registry test_registry_ ## name(#name, &name);  \
    static void name(testing & t)


//
// mtmd_image
//

MAKE_TEST(test_image_preprocessor_lfm2) {
    clip_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);

    // { image size, expected tiling }
    const std::vector<std::pair<clip_image_size, bool>> cases = {
        { {  704, 704 }, false },
        // 720 / (patch_size * n_merge) is exactly 22.5, so this only matches HF
        // if round_by_factor rounds half to even (22) instead of away from zero (23)
        { {  720, 720 }, false },
        { {  736, 736 }, true  },
        { { 1024, 977 }, true  },
        { { 1056, 384 }, false },
    };

    for (const auto & [size, expected] : cases) {
        const bool actual = mtmd_image_preprocessor_lfm2::should_tile(hparams, size);

        t.assert_equal(
            "tiling for " + std::to_string(size.width) + "x" + std::to_string(size.height),
            std::string(expected ? "tiled" : "single"),
            std::string(actual   ? "tiled" : "single"));
    }
}

MAKE_TEST(test_image_preprocessor_deepseek4_vision) {
    clip_hparams hparams;
    hparams.patch_size = 14;
    hparams.n_merge = 3;
    hparams.image_min_pixels = 147456;
    hparams.image_max_tokens = 384;
    hparams.image_max_wh_ratio = 8;
    hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
    hparams.image_pad_color = {127, 127, 127};
    for (int i = 0; i < 3; i++) {
        hparams.image_mean[i] = 0.5f;
        hparams.image_std[i] = 0.5f;
    }

    struct test_case {
        const char * name;
        int width;
        int height;
        int expected_width;
        int expected_height;
        uint64_t expected_hash;
    };
    const std::vector<test_case> cases = {
        {"square", 384, 384, 392, 392, 15370074496759765159ULL},
        {"tiny",    16,  16, 392, 392,  4444667534160286557ULL},
        {"lt8",    700, 100, 1022, 154, 7029455861156325712ULL},
        {"eq8",    800, 100, 1092, 140, 7176349139487537531ULL},
        {"gt8",    900, 100, 1092, 140, 16373669385831837058ULL},
        {"tall",   100, 700, 154, 1022, 3769855483192370884ULL},
        {"odd",    350, 490, 350, 490, 16201472341062198551ULL},
        {"even",   490, 350, 490, 350, 15802083030392099821ULL},
    };

    mtmd_image_preprocessor_deepseek4_vision preprocessor(hparams);
    for (const auto & tc : cases) {
        clip_image_u8 image;
        image.set_size({tc.width, tc.height}, false);
        std::vector<uint8_t> pixels((size_t) tc.width * tc.height * 3);
        for (int y = 0; y < tc.height; y++) {
            for (int x = 0; x < tc.width; x++) {
                const size_t off = ((size_t) y * tc.width + x) * 3;
                pixels[off + 0] = (x * 17 + y * 3) % 256;
                pixels[off + 1] = (x * 5 + y * 11) % 256;
                pixels[off + 2] = (x * 13 + y * 7) % 256;
            }
        }
        image.cpy_buf(pixels);

        auto output = preprocessor.preprocess(image);
        t.assert_equal(tc.name + std::string(" entry count"), 1, (int) output.entries.size());
        const auto & entry = output.entries[0];
        t.assert_equal(tc.name + std::string(" width"), tc.expected_width, entry.nx());
        t.assert_equal(tc.name + std::string(" height"), tc.expected_height, entry.ny());

        uint64_t hash = 1469598103934665603ULL;
        for (float value : entry.get_ro_buf()) {
            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            for (int i = 0; i < 4; i++) {
                hash ^= (bits >> (8 * i)) & 0xff;
                hash *= 1099511628211ULL;
            }
        }
        t.assert_equal(tc.name + std::string(" pixels"), std::to_string(tc.expected_hash), std::to_string(hash));
    }
}

//
// mtmd temporal merge
//

MAKE_TEST(test_temporal_merge_grouping) {
    std::vector<mtmd::bitmap_ptr> pool; // keeps the bitmaps alive until the end of the test

    // spec chars:
    //   v = video frame, w = video frame of another size, a = audio, i = plain image, t = text
    auto make_parts = [&pool](const std::string & spec) {
        std::vector<mtmd_input_part> parts;
        for (char c : spec) {
            if (c == 't') {
                parts.push_back({ "hello", nullptr });
                continue;
            }
            mtmd_bitmap * bm = nullptr;
            switch (c) {
                case 'v': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                case 'w': bm = mtmd_bitmap_init(200, 200, nullptr);   break;
                case 'a': bm = mtmd_bitmap_init_from_audio(100, nullptr); break;
                case 'i': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                default: throw std::runtime_error(std::string("unknown spec char: ") + c);
            }
            mtmd_bitmap_set_mergeable(bm, c != 'i');
            pool.emplace_back(bm);
            parts.push_back({ "", bm });
        }
        return parts;
    };

    // { parts, n_merge, expected size of each group }
    const std::vector<std::tuple<std::string, int, std::string>> cases = {
        { "vv",   2, "2"    },
        { "vvv",  2, "21"   },
        { "vvvv", 2, "22"   },
        { "vvi",  2, "21"   },
        { "tvvt", 2, "2"    },
        { "vtv",  2, "11"   }, // text in between breaks the merge
        { "vw",   2, "11"   }, // different sizes cannot be merged
        { "aa",   2, "11"   }, // audio is never merged
        { "ii",   2, "11"   }, // two unrelated images must stay separated
        { "iv",   2, "11"   },
        { "vi",   2, "11"   },
        { "vv",   1, "11"   }, // model without temporal merge
    };

    for (const auto & [spec, n_merge, expected] : cases) {
        auto parts  = make_parts(spec);
        auto groups = mtmd_group_mergeable_bitmaps(parts, n_merge);

        std::string actual;
        for (const auto & group : groups) {
            actual += std::to_string(group.size());
        }

        const std::string name = "\"" + spec + "\" with n_merge=" + std::to_string(n_merge);
        t.assert_equal("groups for " + name, expected, actual);

        size_t n_bitmap_parts = 0;
        for (const auto & p : parts) {
            n_bitmap_parts += p.bitmap != nullptr ? 1 : 0;
        }
        t.assert_equal("remaining bitmap parts for " + name, groups.size(), n_bitmap_parts);
    }
}

//
// main
//

int main(int argc, char ** argv) {
    testing t(std::cout);
    t.verbose = true;

    // usage: test-mtmd-impl [filter_regex]
    for (int i = 1; i < argc; i++) {
        t.set_filter(argv[i]);
    }

    for (const auto & e : test_registry::all()) {
        t.test(e.name, e.fn);
    }

    return t.summary();
}
