// Exercise the production enqueue helper with deterministic queue failures.
// No torch/CANN libraries or device are needed: only Runtime's boundary is fake.
#include "stable/exec.h"

#include <iostream>
#include <thread>

using fla_npu_stable::Runtime;
using fla_npu_stable::stable::aclOpExecutor;
namespace launch_detail = fla_npu_stable::stable::detail;

namespace {
constexpr const char* kRefused =
    "External NPU stream is not supported by task queue enqueue.";
std::string scenario;
int launches = 0;
int enqueues = 0;
int destroyed = 0;
std::function<int()> pending;

struct QueueError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::logic_error(message);
  }
}

struct Descriptor {
  ~Descriptor() { ++destroyed; }
};

int launch(void*, uint64_t, aclOpExecutor*, void*) {
  require(destroyed == 0, "descriptor was released before launch");
  require(++launches == 1, "executor was launched twice");
  if (scenario == "callback_throws") {
    // Even a matching refusal message is unsafe once the callback has started.
    throw QueueError(kRefused);
  }
  return scenario == "launch_status" ? 17 : 0;
}

void submit() {
  auto held = std::make_tuple(std::make_unique<Descriptor>());
  if (!launch_detail::enqueue_launch(Runtime::instance(), "aclnnTest", launch,
                                     0, nullptr, 0, nullptr, held)) {
    require(std::get<0>(held) != nullptr, "fallback lost its descriptors");
    launch(nullptr, 0, nullptr, nullptr);
  }
}
}  // namespace

namespace fla_npu_stable {
Runtime& Runtime::instance() {
  static Runtime runtime;
  return runtime;
}

bool Runtime::enqueue_enabled() { return scenario != "queue_disabled"; }
std::string Runtime::recent_error_message() { return "test launch error"; }

void Runtime::enqueue(const std::string&, const std::function<int()>& fn) {
  ++enqueues;
  if (scenario == "refused") {
    throw QueueError(std::string(kRefused) + "\nException raised from enCurrentNPUStream");
  }
  if (scenario == "unknown_before") {
    throw QueueError("unknown enqueue failure");
  }
  if (scenario == "deferred" || scenario == "accepted_then_throws") {
    pending = fn;
    if (scenario == "accepted_then_throws") {
      throw QueueError("failure after queue accepted the callback");
    }
    return;
  }
  if (scenario == "worker_then_throws") {
    std::thread worker([&] { fn(); });
    worker.join();
  } else {
    fn();
  }
  if (scenario == "sync_throws" || scenario == "worker_then_throws") {
    throw QueueError("synchronize failed: 507057");
  }
  if (scenario == "refusal_after_callback") {
    throw QueueError(kRefused);
  }
}
}  // namespace fla_npu_stable

int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected one scenario");
    scenario = argv[1];
    std::string expected_error;
    if (scenario == "unknown_before") expected_error = "unknown enqueue failure";
    if (scenario == "sync_throws" || scenario == "worker_then_throws") {
      expected_error = "synchronize failed: 507057";
    }
    if (scenario == "callback_throws" || scenario == "refusal_after_callback") {
      expected_error = kRefused;
    }
    if (scenario == "accepted_then_throws") {
      expected_error = "failure after queue accepted the callback";
    }

    bool caught = false;
    try {
      submit();
    } catch (const QueueError& error) {
      caught = true;
      require(error.what() == expected_error, "original exception was replaced");
    }
    require(caught == !expected_error.empty(), "exception was swallowed");
    require(enqueues == (scenario == "queue_disabled" ? 0 : 1), "wrong enqueue count");
    if (pending) {
      require(launches == 0 && destroyed == 0, "pending launch lost ownership");
      pending();
      pending = nullptr;
    }
    require(launches == (scenario == "unknown_before" ? 0 : 1), "wrong launch count");
    require(destroyed == 1, "descriptor must be destroyed exactly once");
    if (scenario == "launch_status") {
      bool reported = false;
      try {
        launch_detail::check_async_failure();
      } catch (const std::runtime_error& error) {
        reported = std::string(error.what()).find(
                       "aclnnTest failed asynchronously: 17") != std::string::npos;
      }
      require(reported, "asynchronous status was lost");
      launch_detail::check_async_failure();  // the status is consumed exactly once
    }
    std::cout << scenario << ": PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << scenario << ": " << error.what() << '\n';
    return 1;
  }
}
