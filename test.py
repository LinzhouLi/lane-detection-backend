import subprocess
import os


def runcmd(command):
    ret = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
                         timeout=10)
    if ret.returncode == 0:
        print("success:", ret.stdout)
    else:
        print("error:", ret.stderr)


def pred_image(file_path):
    extension = file_path.rsplit('.', 1)[-1]
    prefix_folder = file_path.rsplit('\\', 1)[1].rsplit('.', 1)[0]
    print(extension, prefix_folder)
    commands = (
            "conda activate " + "lane" + "\n" +
            "python tools/vis/lane_img_dir.py"
            " --image-path=" + os.path.join("image", prefix_folder) +
            " --image-suffix=" + extension +
            " --save-path=" + "image-pred" +
            " --pred --config=configs/lane_detection/bezierlanenet/resnet18_culane-aug1b.py" +
            " --checkpoint=checkpoints/trained_model/resnet18_bezierlanenet_culane_aug1b_20211109.pt\n"
    )
    process = subprocess.Popen("cmd.exe", shell=False, universal_newlines=True,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdin, stdout = process.communicate(commands)
    print("input:\n", stdin)
    print("output:\n", stdout)
    return


pred_image("image\\00020\\00020.jpg")
