#include "FHE_anonymous.h"

#define VERBOSE 1

using namespace troy;
using namespace troy::linear;
using std::stringstream;
using std::vector;
using namespace std;

int main(int argc, char **argv)
{
    int port = atoi(argv[1]);
    const char *image_path;
    int truth_label;
    std::string mode;

    if (argc < 3)
    {
        printf("Use default image airplane.png\n");
        image_path = "../airplane.png";
        truth_label = 0;
    }
    else
    {
        image_path = argv[1];
        truth_label = atoi(argv[2]);
        mode = argv[3];
    }

    int socket_fd = init_client(port);

    size_t poly_modulus_degree = 4096;
    // size_t poly_modulus_degree = 16384;
    // size_t poly_modulus_degree = 32768;
    EncryptionParameters parms(SchemeType::CKKS);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::create(poly_modulus_degree, {50, 50}));

    auto context = HeContext::create(parms, true, SecurityLevel::Classical128);
    print_parameters(*context);
    std::cout << std::endl;

    static CKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    std::cout << "Number of slots: " << slot_count << std::endl;

    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key = keygen.create_public_key(false);
    RelinKeys relin_keys = keygen.create_relin_keys(false);
    GaloisKeys gal_keys = keygen.create_galois_keys(false);
    Encryptor encryptor(context);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    encryptor.set_public_key(public_key);

    stringstream pk_stream;
    public_key.save(pk_stream, context);
    sendStream(socket_fd, pk_stream);

    stringstream sk_stream;
    secret_key.save(sk_stream);
    sendStream(socket_fd, sk_stream);



    double scale = (1 << 20);

    if (utils::device_count() > 1)
    {
        std::cerr << "Move to device" << std::endl;
        context->to_device_inplace();
        encoder.to_device_inplace();
        keygen.to_device_inplace();
        encryptor.to_device_inplace();
        decryptor.to_device_inplace();
    }

    ResnetContext resnet_context = {parms, context, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, gal_keys};
    resnet_context.role = Role::Client;
    resnet_context.socket_fd = socket_fd;

    vector<double> input_image = read_image(image_path);
    vector<double> padded_image = pad_image(input_image, 32, 32, 3, 1);

    sendVector(resnet_context.socket_fd, padded_image);

    auto start_time = std::chrono::high_resolution_clock::now();

    

    std::cout << "========== init layer ================" << std::endl;

    Conv2dHelper helper_3_16_3_3_1_1 = Conv2dHelper(1, 3, 16, 32 + 2, 32 + 2, 3, 3, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft);

    Conv2dHelper help_16_16_3_1_1 = Conv2dHelper(1, 16, 16, 32 + 2, 32 + 2, 3, 3, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft);

    Cipher2d empty;

    resnet_context.relu_anonymous_interact(empty, encoder, decryptor, helper_3_16_3_3_1_1, help_16_16_3_1_1, 32, 32, 16, 1, true);

    std::cout << "========== layer 1 ================" << std::endl;

    std::cout << "=== block 0 ===" << std::endl;

    resnet_context.relu_anonymous_interact(empty, encoder, decryptor, help_16_16_3_1_1, help_16_16_3_1_1, 32, 32, 16, 1);

    resnet_context.relu_anonymous_interact_add(empty, encoder, decryptor, help_16_16_3_1_1, help_16_16_3_1_1, 32, 32, 16, 1, true);
    
    std::cout << "=== block 1 ===" << std::endl;

    resnet_context.relu_anonymous_interact(empty, encoder, decryptor, help_16_16_3_1_1, help_16_16_3_1_1, 32, 32, 16, 1);

    resnet_context.relu_anonymous_interact_add(empty, encoder, decryptor, help_16_16_3_1_1, help_16_16_3_1_1, 32, 32, 16, 1, true);

    std::cout << "=== block 2 ===" << std::endl;

    resnet_context.relu_anonymous_interact(empty, encoder, decryptor, help_16_16_3_1_1, help_16_16_3_1_1, 32, 32, 16, 1);

    Conv2dHelper help_16_32_3_2_1 = Conv2dHelper(1, 16, 32, 32 + 2, 32 + 2, 3, 3, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft); // stride = 2 
    Conv2dHelper help_16_32_1_2_0 = Conv2dHelper(1, 16, 32, 32, 32, 1, 1, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft);

    resnet_context.relu_anonymous_interact_add_double(empty, empty, encoder, decryptor, help_16_16_3_1_1, help_16_32_3_2_1, help_16_32_1_2_0,  32, 32, 16, 1, true);

    
    std::cout << "========== layer 2 ================" << std::endl;

    std::cout << "=== block 0 ===" << std::endl;

    Conv2dHelper help_32_32_3_1_1 = Conv2dHelper(1, 32, 32, 16 + 2, 16 + 2, 3, 3, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft);
    resnet_context.relu_anonymous_interact_downsample(empty, encoder, decryptor, help_16_32_3_2_1, help_32_32_3_1_1, 16, 16, 32, 1);

    resnet_context.relu_anonymous_interact_downsample_add(empty, empty, encoder, decryptor, help_32_32_3_1_1, help_16_32_1_2_0, help_32_32_3_1_1, 16, 16, 32, 1, true); //

    std::cout << "=== block 1 ===" << std::endl;

    resnet_context.relu_anonymous_interact(empty, encoder, decryptor, help_32_32_3_1_1, help_32_32_3_1_1, 16, 16, 32, 1);
    resnet_context.relu_anonymous_interact_add(empty, encoder, decryptor, help_32_32_3_1_1, help_32_32_3_1_1, 16, 16, 32, 1, true); 

    std::cout << "=== block 2 ===" << std::endl;

    resnet_context.relu_anonymous_interact(empty, encoder, decryptor, help_32_32_3_1_1, help_32_32_3_1_1, 16, 16, 32, 1);

    Conv2dHelper help_32_64_3_2_1 = Conv2dHelper(1, 32, 64, 16 + 2, 16 + 2, 3, 3, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft);
    Conv2dHelper help_32_64_1_2_0 = Conv2dHelper(1, 32, 64, 16, 16, 1, 1, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft);
    resnet_context.relu_anonymous_interact_add_double(empty, empty, encoder, decryptor, help_32_32_3_1_1, help_32_64_3_2_1, help_32_64_1_2_0,  16, 16, 32, 1, true);

    std::cout << "========== layer 3 ================" << std::endl;

    std::cout << "=== block 0 ===" << std::endl;

    Conv2dHelper help_64_64_3_1_1 = Conv2dHelper(1, 64, 64, 8 + 2, 8 + 2, 3, 3, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft);
    resnet_context.relu_anonymous_interact_downsample(empty, encoder, decryptor, help_32_64_3_2_1, help_64_64_3_1_1, 8, 8, 64, 1);
    resnet_context.relu_anonymous_interact_downsample_add(empty, empty, encoder, decryptor, help_64_64_3_1_1, help_32_64_1_2_0, help_64_64_3_1_1, 8, 8, 64, 1, true);

    std::cout << "=== block 1 ===" << std::endl;

    resnet_context.relu_anonymous_interact(empty, encoder, decryptor, help_64_64_3_1_1, help_64_64_3_1_1, 8, 8, 64, 1);
    resnet_context.relu_anonymous_interact_add(empty, encoder, decryptor, help_64_64_3_1_1, help_64_64_3_1_1, 8, 8, 64, 1, true);

    std::cout << "=== block 2 ===" << std::endl;

    resnet_context.relu_anonymous_interact(empty, encoder, decryptor, help_64_64_3_1_1, help_64_64_3_1_1, 8, 8, 64, 1);
    // resnet_context.relu_anonymous_interact_add(empty, encoder, decryptor, help_64_64_3_1_1, help_64_64_3_1_1, 8, 8, 64, 1, true);

    std::cout << "========== Average Pooling ================" << std::endl;

    MatmulHelper mat_help_10_64_1 = MatmulHelper(10, 64, 1, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);

    resnet_context.relu_anonymous_interact_add_avgpool(empty, encoder, decryptor, help_64_64_3_1_1, mat_help_10_64_1, 8, 8, 64, 1, true);




    print_socket_data_stats();
    close(socket_fd);

    return 0;
}
