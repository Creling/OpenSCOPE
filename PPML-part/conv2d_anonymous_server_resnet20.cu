#include "FHE_anonymous.h"


#define MEASURE_TIME(expression)                                                                             \
    [&]() -> decltype(auto) {                                                                                \
        auto start = std::chrono::high_resolution_clock::now();                                              \
        auto result = (expression);                                                                          \
        auto end = std::chrono::high_resolution_clock::now();                                                \
        std::chrono::duration<double> duration = end - start;                                                \
        std::cout << "Execution time of " #expression ": \t" << duration.count() << " seconds" << std::endl; \
        return result;                                                                                       \
    }()

#define PRINT_2D_VECTOR_SHAPE(vec) \
    printf("%d\t" #vec ".shape: (%lu, %lu)\n", __LINE__, vec.size(), vec[0].size())

#define VERBOSE 1

using namespace troy;
using namespace troy::linear;
using std::stringstream;
using std::vector;
using namespace std;

int main(int argc, char **argv)
{
    int port = atoi(argv[1]);
    int socket_fd = init_server(port);

    size_t poly_modulus_degree = 4096;
    // size_t poly_modulus_degree = 16384;
    // size_t poly_modulus_degree = 32768;

    EncryptionParameters parms(SchemeType::CKKS);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::create(poly_modulus_degree, {50, 50}));
    auto context = HeContext::create(parms, true, SecurityLevel::Classical128);
    print_parameters(*context);


    stringstream pk_stream = receiveStream(socket_fd);
    PublicKey public_key;
    public_key.load(pk_stream, context);

	    stringstream sk_stream = receiveStream(socket_fd);
	    SecretKey secret_key;
	    secret_key.load(sk_stream);

	    RelinKeys relin_keys;
	    GaloisKeys gal_keys;

	    static CKKSEncoder encoder(context);
	    size_t slot_count = encoder.slot_count();
	    std::cout << "Number of slots: " << slot_count << std::endl;

	    Encryptor encryptor(context);
	    encryptor.set_public_key(public_key);
	    Evaluator evaluator(context);
	    Decryptor decryptor(context, secret_key);
	    double scale = (1 << 20);

	    if (utils::device_count() > 0)
    {
        std::cerr << "Move to device" << std::endl;
        context->to_device_inplace();
        encoder.to_device_inplace();
        encryptor.to_device_inplace();
        decryptor.to_device_inplace();
    }

    ResnetContext resnet_context = {parms, context, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, gal_keys};
    resnet_context.role = Role::Server;
    resnet_context.socket_fd = socket_fd;

    auto start_time = std::chrono::high_resolution_clock::now();

    vector<double> padded_image = receiveVector(resnet_context.socket_fd);


    std::cout << "========== init layer ================" << std::endl;

    vector<double> weight = read_file("../weights/model_init_conv1_fused.txt");
    vector<double> bias = read_file("../weights/model_init_bn1_fused.txt");

    Conv2dHelper helper_3_16_3_1_1 = Conv2dHelper(1, 3, 16, 32 + 2, 32 + 2, 3, 3, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft);

    Plain2d in_encoded = helper_3_16_3_1_1.encode_inputs_doubles(encoder, padded_image.data(), std::nullopt, scale);
    Plain2d bias_encoded = helper_3_16_3_1_1.encode_outputs_doubles(encoder, bias.data(), std::nullopt, scale * scale);
    Plain2d weight_encoded = helper_3_16_3_1_1.encode_weights_doubles(encoder, weight.data(), std::nullopt, scale); // std::nullopt means encode to level 0
    Cipher2d in_encrypted = in_encoded.encrypt_asymmetric(encryptor);

    Cipher2d ret = helper_3_16_3_1_1.conv2d(evaluator, in_encrypted, weight_encoded);
    ret.add_plain_inplace(evaluator, bias_encoded);

    vector<double> mask = generate_random_vector(bias.size(), 0.0, 0.1);
    Plain2d mask_encoded = helper_3_16_3_1_1.encode_outputs_doubles(encoder, mask.data(), std::nullopt, scale * scale);

    // y_encrypted.add_plain_inplace(evaluator, mask_encoded);

    // auto ret_decrypted = helper_3_16_3_1_1.decrypt_outputs_doubles(encoder, decryptor, ret);

    Conv2dHelper help_16_16_3_1_1 = Conv2dHelper(1, 16, 16, 32 + 2, 32 + 2, 3, 3, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft);

    ret = resnet_context.relu_anonymous_interact(ret, encoder, decryptor, helper_3_16_3_1_1, help_16_16_3_1_1, 32, 32, 16, 1, true);

    std::cout << "========== layer 1 ================" << std::endl;

    std::cout << "=== block 0 ===" << std::endl;

    ret = resnet_context.Conv2d("../weights/model_layer1_block0_convbn1.txt", "../weights/model_layer1_block0_bias1.txt", ret, help_16_16_3_1_1);

    // ret_decrypted = help_16_16_3_1_1.decrypt_outputs_doubles(encoder, decryptor, ret);

    ret = resnet_context.relu_anonymous_interact(ret, encoder, decryptor, help_16_16_3_1_1, help_16_16_3_1_1, 32, 32, 16, 1);

    ret = resnet_context.Conv2d("../weights/model_layer1_block0_convbn2.txt", "../weights/model_layer1_block0_bias2.txt", ret, help_16_16_3_1_1);


    std::cout << "=== block 1 ===" << std::endl;

    ret = resnet_context.Conv2d("../weights/model_layer1_block1_convbn1.txt", "../weights/model_layer1_block1_bias1.txt", ret, help_16_16_3_1_1);

    // ret_decrypted = help_16_16_3_1_1.decrypt_outputs_doubles(encoder, decryptor, ret);

    ret = resnet_context.relu_anonymous_interact(ret, encoder, decryptor, help_16_16_3_1_1, help_16_16_3_1_1, 32, 32, 16, 1);

    ret = resnet_context.Conv2d("../weights/model_layer1_block1_convbn2.txt", "../weights/model_layer1_block1_bias2.txt", ret, help_16_16_3_1_1);

    ret = resnet_context.relu_anonymous_interact_add(ret, encoder, decryptor, help_16_16_3_1_1, help_16_16_3_1_1, 32, 32, 16, 1); 

    std::cout << "=== block 2 ===" << std::endl;

    ret = resnet_context.Conv2d("../weights/model_layer1_block2_convbn1.txt", "../weights/model_layer1_block2_bias1.txt", ret, help_16_16_3_1_1);

    ret = resnet_context.relu_anonymous_interact(ret, encoder, decryptor, help_16_16_3_1_1, help_16_16_3_1_1, 32, 32, 16, 1);

    ret = resnet_context.Conv2d("../weights/model_layer1_block2_convbn2.txt", "../weights/model_layer1_block2_bias2.txt", ret, help_16_16_3_1_1);

    Conv2dHelper help_16_32_3_2_1 = Conv2dHelper(1, 16, 32, 32 + 2, 32 + 2, 3, 3, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft); // stride = 2
    Cipher2d ret2;
    Conv2dHelper help_16_32_1_2_0 = Conv2dHelper(1, 16, 32, 32, 32, 1, 1, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft);
    ret = resnet_context.relu_anonymous_interact_add_double(ret, ret2, encoder, decryptor, help_16_16_3_1_1, help_16_32_3_2_1, help_16_32_1_2_0,  32, 32, 16, 1, true);

    std::cout << "========== layer 2 ================" << std::endl;

    std::cout << "=== block 0 ===" << std::endl;
    Conv2dHelper help_32_32_3_1_1 = Conv2dHelper(1, 32, 32, 16 + 2, 16 + 2, 3, 3, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft);

    ret = resnet_context.relu_anonymous_interact_downsample(ret, encoder, decryptor, help_16_32_3_2_1, help_32_32_3_1_1, 16, 16, 32, 1);
    ret = resnet_context.Conv2d("../weights/model_layer2_block0_convbn2.txt", "../weights/model_layer2_block0_bias2.txt", ret, help_32_32_3_1_1); // 8192
    // ret_decrypted = help_32_32_3_1_1.decrypt_outputs_doubles(encoder, decryptor, ret);

    ret2 = resnet_context.Conv2d("../weights/model_layer2_block0_convbn3.txt", "../weights/model_layer2_block0_bias3.txt", ret2, help_16_32_1_2_0); // 32768
    // auto ret2_decrypted = help_16_32_1_2_0.decrypt_outputs_doubles(encoder, decryptor, ret2);

    ret = resnet_context.relu_anonymous_interact_downsample_add(ret, ret2, encoder, decryptor, help_32_32_3_1_1, help_16_32_1_2_0, help_32_32_3_1_1, 16, 16, 32, 1, true); //
    
    std::cout << "=== block 1 ===" << std::endl;

    ret = resnet_context.Conv2d("../weights/model_layer2_block1_convbn1.txt", "../weights/model_layer2_block1_bias1.txt", ret, help_32_32_3_1_1);
    // ret_decrypted = help_32_32_3_1_1.decrypt_outputs_doubles(encoder, decryptor, ret);
    ret = resnet_context.relu_anonymous_interact(ret, encoder, decryptor, help_32_32_3_1_1, help_32_32_3_1_1, 16, 16, 32, 1);
    ret = resnet_context.Conv2d("../weights/model_layer2_block1_convbn2.txt", "../weights/model_layer2_block1_bias2.txt", ret, help_32_32_3_1_1);
    // ret_decrypted = help_32_32_3_1_1.decrypt_outputs_doubles(encoder, decryptor, ret);

    ret = resnet_context.relu_anonymous_interact_add(ret, encoder, decryptor, help_32_32_3_1_1, help_32_32_3_1_1, 16, 16, 32, 1, true); 

    std::cout << "=== block 2 ===" << std::endl;

    ret = resnet_context.Conv2d("../weights/model_layer2_block2_convbn1.txt", "../weights/model_layer2_block2_bias1.txt", ret, help_32_32_3_1_1);
    ret = resnet_context.relu_anonymous_interact(ret, encoder, decryptor, help_32_32_3_1_1, help_32_32_3_1_1, 16, 16, 32, 1);
    ret = resnet_context.Conv2d("../weights/model_layer2_block2_convbn2.txt", "../weights/model_layer2_block2_bias2.txt", ret, help_32_32_3_1_1);
    // ret_decrypted = help_32_32_3_1_1.decrypt_outputs_doubles(encoder, decryptor, ret);

    Conv2dHelper help_32_64_3_2_1 = Conv2dHelper(1, 32, 64, 16 + 2, 16 + 2, 3, 3, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft);
    Conv2dHelper help_32_64_1_2_0 = Conv2dHelper(1, 32, 64, 16, 16, 1, 1, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft);
    ret = resnet_context.relu_anonymous_interact_add_double(ret, ret2, encoder, decryptor, help_32_32_3_1_1, help_32_64_3_2_1, help_32_64_1_2_0,  16, 16, 32, 1, true);


    std::cout << "========== layer 3 ================" << std::endl;

    std::cout << "=== block 0 ===" << std::endl;

    // ret_decrypted = help_32_64_3_2_1.decrypt_outputs_doubles(encoder, decryptor, ret);
    Conv2dHelper help_64_64_3_1_1 = Conv2dHelper(1, 64, 64, 8 + 2, 8 + 2, 3, 3, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft);
    ret = resnet_context.relu_anonymous_interact_downsample(ret, encoder, decryptor, help_32_64_3_2_1, help_64_64_3_1_1, 8, 8, 64, 1);
    ret = resnet_context.Conv2d("../weights/model_layer3_block0_convbn2.txt", "../weights/model_layer3_block0_bias2.txt", ret, help_64_64_3_1_1);
    // ret_decrypted = help_64_64_3_1_1.decrypt_outputs_doubles(encoder, decryptor, ret);

    // downsample
    ret2 = resnet_context.Conv2d("../weights/model_layer3_block0_convbn3.txt", "../weights/model_layer3_block0_bias3.txt", ret2, help_32_64_1_2_0);
    // ret2_decrypted = help_32_64_1_2_0.decrypt_outputs_doubles(encoder, decryptor, ret2);
    ret = resnet_context.relu_anonymous_interact_downsample_add(ret, ret2, encoder, decryptor, help_64_64_3_1_1, help_32_64_1_2_0, help_64_64_3_1_1, 8, 8, 64, 1, true);

    std:: cout << "=== block 1 ===" << std::endl;

    ret = resnet_context.Conv2d("../weights/model_layer3_block1_convbn1.txt", "../weights/model_layer3_block1_bias1.txt", ret, help_64_64_3_1_1);
    // ret_decrypted = help_64_64_3_1_1.decrypt_outputs_doubles(encoder, decryptor, ret);
    ret = resnet_context.relu_anonymous_interact(ret, encoder, decryptor, help_64_64_3_1_1, help_64_64_3_1_1, 8, 8, 64, 1);
    ret = resnet_context.Conv2d("../weights/model_layer3_block1_convbn2.txt", "../weights/model_layer3_block1_bias2.txt", ret, help_64_64_3_1_1);
    // writeToFile(ret_decrypted, "conv2d_anonymous_im2col_layer3_block1_convbn2.txt"); 
    ret = resnet_context.relu_anonymous_interact_add(ret, encoder, decryptor, help_64_64_3_1_1, help_64_64_3_1_1, 8, 8, 64, 1, true);

    std::cout << "=== block 2 ===" << std::endl;

    ret = resnet_context.Conv2d("../weights/model_layer3_block2_convbn1.txt", "../weights/model_layer3_block2_bias1.txt", ret, help_64_64_3_1_1);
    ret = resnet_context.relu_anonymous_interact(ret, encoder, decryptor, help_64_64_3_1_1, help_64_64_3_1_1, 8, 8, 64, 1);
    ret = resnet_context.Conv2d("../weights/model_layer3_block2_convbn2.txt", "../weights/model_layer3_block2_bias2.txt", ret, help_64_64_3_1_1);


    std::cout << "========== FC AVG ================" << std::endl;

    MatmulHelper mat_help_10_64_1 = MatmulHelper(10, 64, 1, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);
    ret = resnet_context.relu_anonymous_interact_add_avgpool(ret, encoder, decryptor, help_64_64_3_1_1, mat_help_10_64_1, 8, 8, 64, 1, true);

    weight = read_file("../weights/model_fc_weight.txt");
    bias = read_file("../weights/model_fc_bias.txt");

    weight_encoded = mat_help_10_64_1.encode_inputs_doubles(encoder, weight.data(), std::nullopt, scale);
    bias_encoded = mat_help_10_64_1.encode_outputs_doubles(encoder, bias.data(), std::nullopt, scale * scale);
    ret = mat_help_10_64_1.matmul_reverse(evaluator, weight_encoded, ret);

    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Total execution time: " << duration << " ms" << std::endl;

    print_socket_data_stats();
    close(socket_fd);

    return 0;

}
