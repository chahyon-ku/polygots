import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, default='preds/best/')
    parser.add_argument('--output_path', type=str, default='best_ner_results.txt')
    args = parser.parse_args()

    # output
    with open(args.output_path, 'w', encoding='utf-8') as f:
        rows = []
        for root, dirs, files in os.walk(args.pred_dir):
            for file in files:
                if file.endswith('.eval'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f2:
                        result = json.load(f2)
                    row = '{}, {:.4f}, {:.4f}, {:.4f}'.format(os.join(root, file).replace(args.pred_dir, ''), result['f1'], result['p'], result['r'])
                    rows.append(row)
        
        f.write('name, f1, p, r\n')
        rows = sorted(rows, key=lambda x: float(x.split(',')[0]), reverse=True)
        for row in rows:
            f.write(row + '\n')


if __name__ == '__main__':
    main()