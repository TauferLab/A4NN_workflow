from torch import package
import os

def model_package(model, path, arch_id, epoch):
    save_path = os.path.join(path, "{}_epoch_{}.pt".format(arch_id, epoch))
    package_name = "{}_epoch_{}".format(arch_id, epoch)
    resource_name = "{}_epoch_{}.pkl".format(arch_id, epoch)

    with package.PackageExporter(save_path) as exporter:
        exporter.extern("numpy.**")
        exporter.intern("nsganet.models.**")
        exporter.intern("nas_search.**")
        
        exporter.save_pickle(package_name, resource_name, model)
