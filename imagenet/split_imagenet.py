import os, argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data-path', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    os.makedirs(os.path.join(args.data_path, 'labeled'), exist_ok=True)

    with open('./imagenet_split/train_labeled.txt', 'r') as f:
        lines = f.readlines()
        f.close()

    for l in lines:
        image_name = l.split()[0]
        folder, _ = image_name.split('/')

        cmd = "mkdir -p %s && cp %s %s" % (
                os.path.join(args.data_path, "labeled", folder),
                os.path.join(args.data_path, "train", image_name),
                os.path.join(args.data_path, "labeled", image_name)
        )
        print(cmd)
        os.system(cmd)

    os.makedirs(os.path.join(args.data_path, 'unlabeled'), exist_ok=True)

    with open('./imagenet_split/train_unlabeled.txt', 'r') as f:
        lines = f.readlines()
        f.close()

    for l in lines:
        image_name = l.split()[0]
        folder, _ = image_name.split('/')

        cmd = "mkdir -p %s && cp %s %s" % (
                os.path.join(args.data_path, "unlabeled", folder),
                os.path.join(args.data_path, "train", image_name),
                os.path.join(args.data_path, "unlabeled", image_name)
        )
        print(cmd)
        os.system(cmd)
