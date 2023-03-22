set(TARGET main)
add_executable(${TARGET}
               main.cpp
               run.cpp)

if(NOT WIN32)
    target_sources(${TARGET} PRIVATE tcp_server.cpp)
endif()

target_link_libraries(${TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
