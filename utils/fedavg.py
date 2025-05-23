def fed_avg(model_list):
    avg_model = model_list[0]
    for key in avg_model.state_dict().keys():
        for i in range(1, len(model_list)):
            avg_model.state_dict()[key] += model_list[i].state_dict()[key]
        avg_model.state_dict()[key] = avg_model.state_dict()[key] / len(model_list)
    return avg_model
