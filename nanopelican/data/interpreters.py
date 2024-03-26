def null_func(x, y):
    return x, y


def get_interpreter(data_interpreter):
    my_dict = {
        'zenodo': zenodo
    }

    if type(data_interpreter) == str and data_interpreter in my_dict:
        return my_dict[data_interpreter]
    print(f"Could not find interpreter: {data_interpreter}, using default")
    return null_func

def zenodo(feature_data, label_data):
    # jetconstituents = file.get('jetConstituentList')
    # Shape: num_jets x num_particles x 4
    fourvectors = feature_data[:, :, :4] # Particles (px, py, pz, E)
    target = label_data[:, -6:-1]
    return fourvectors, target
