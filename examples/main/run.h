#pragma once

#include "llama.h"
#include "common.h"

int run(llama_context * ctx,
        gpt_params params,
        std::istream & instream,
        FILE *outstream,
        FILE *errstream);
