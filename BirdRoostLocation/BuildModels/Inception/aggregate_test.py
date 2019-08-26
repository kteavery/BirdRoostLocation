import os.path
from BirdRoostLocation import utils
import BirdRoostLocation.LoadSettings as settings
from BirdRoostLocation.BuildModels.Inception import aggregate_train
import os


def test(model, bottleneck_list, radar_fields):
    x, y, _ = aggregate_train.get_random_cached_bottlenecks(
        image_lists=bottleneck_list,
        how_many=-1,
        category="training",
        radar_fields=radar_fields,
    )
    result = model.evaluate(x, y, 32)
    print(result)


def main():
    os.chdir(settings.WORKING_DIRECTORY)
    dual_pol = True

    if dual_pol:
        radar_field = utils.Radar_Products.cc
        radar_fields = aggregate_train.dual_pol_fields
        save = "dual_pol.h5"
        model = aggregate_train.create_model(8192, save)
    else:
        radar_field = utils.Radar_Products.reflectivity
        radar_fields = aggregate_train.legacy_fields
        save = "legacy.h5"
        model = aggregate_train.create_model(4096, save)

    image_lists = aggregate_train.create_image_lists(radar_field)
    bottleneck_list = aggregate_train.get_bottleneck_list(image_lists, radar_field)
    test(model, bottleneck_list, radar_fields)


if __name__ == "__main__":
    main()
