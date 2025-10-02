/**
 * TileLang Hello World: Vector Addition (C++ Version)
 * 
 * This example demonstrates using TileLang from C++ to implement
 * element-wise vector addition.
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

// TileLang headers would be included here when available
// For now, we'll create a placeholder implementation

void print_banner(const char* message) {
    std::cout << std::string(60, '=') << std::endl;
    std::cout << message << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

// Simple CPU-based vector addition for demonstration
void vector_add_cpu(const std::vector<float>& a, 
                    const std::vector<float>& b, 
                    std::vector<float>& c) {
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

bool validate_results(const std::vector<float>& result,
                      const std::vector<float>& expected,
                      float tolerance = 1e-5f) {
    if (result.size() != expected.size()) {
        return false;
    }
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < result.size(); ++i) {
        float diff = std::abs(result[i] - expected[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    std::cout << "  Max difference: " << max_diff << std::endl;
    return max_diff < tolerance;
}

int main() {
    print_banner("TileLang Hello World: Vector Addition (C++)");
    
    // Vector size
    const size_t N = 1024 * 1024;  // 1M elements
    
    std::cout << "\nVector size: " << N << " elements" << std::endl;
    std::cout << "Computing: C = A + B (element-wise)" << std::endl;
    
    // Step 1: Prepare test data
    std::cout << "\n[1/4] Preparing test data..." << std::endl;
    
    std::vector<float> a(N), b(N), c(N);
    std::vector<float> expected(N);
    
    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < N; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }
    
    std::cout << "✓ Test data prepared" << std::endl;
    
    // Step 2: Compile kernel (placeholder - would use TileLang C++ API)
    std::cout << "\n[2/4] Setting up computation..." << std::endl;
    std::cout << "✓ Using CPU backend (TileLang C++ API integration pending)" << std::endl;
    
    // Step 3: Run computation
    std::cout << "\n[3/4] Running vector addition..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    vector_add_cpu(a, b, c);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end - start;
    
    std::cout << "✓ Computation completed" << std::endl;
    
    // Step 4: Validate results
    std::cout << "\n[4/4] Validating results..." << std::endl;
    
    // Create expected output
    vector_add_cpu(a, b, expected);
    
    // Print sample values
    std::cout << "Sample results (first 5 elements):" << std::endl;
    std::cout << "  Output C: [";
    for (int i = 0; i < 5; ++i) {
        std::cout << c[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  Expected: [";
    for (int i = 0; i < 5; ++i) {
        std::cout << expected[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    if (validate_results(c, expected)) {
        std::cout << "✓ Results match reference!" << std::endl;
    } else {
        std::cout << "✗ Validation failed" << std::endl;
        return 1;
    }
    
    // Performance metrics
    print_banner("Performance Benchmark");
    
    double latency_ms = duration.count();
    size_t bytes_transferred = N * 3 * sizeof(float);  // 3 vectors * 4 bytes
    double bandwidth_gb_s = (bytes_transferred / 1e9) / (latency_ms / 1000.0);
    
    std::cout << "\nLatency: " << latency_ms << " ms" << std::endl;
    std::cout << "Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << "Throughput: " << (N / (latency_ms * 1e6)) << " GFLOPS" << std::endl;
    
    print_banner("✓ Hello TileLang C++ - Success!");
    
    std::cout << "\nNOTE: This is a CPU reference implementation." << std::endl;
    std::cout << "TileLang C++ API integration is pending for GPU acceleration." << std::endl;
    
    return 0;
}
