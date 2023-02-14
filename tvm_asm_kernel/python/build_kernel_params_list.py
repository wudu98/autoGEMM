import sys
import os
import json

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, ".."))
    input_file_path = os.path.join(project_dir, f"build/scheduler_summary.log")
    # input_file_path = os.path.join(project_dir, "./cache_block_size.json")
    output_file_path = os.path.join(project_dir, "src/kernel_params_list.hpp")

    if len(sys.argv) >= 2 and sys.argv[1] == 'clean':
        if os.path.exists(output_file_path):
            print("rm %s" % output_file_path)
            os.remove(output_file_path)
    else:
        if not os.path.exists(input_file_path):
            exit(-1)
        
        cc_code = f"""#ifndef __KERNEL_PARAMS_LIST_H_
#define __KERNEL_PARAMS_LIST_H_

#include <list>

namespace KernelParams
{{
    struct SimpleStruct {{
        int M;
        int N;
        int K;
        int nc;
        int kc;
        int padding_size;
        SimpleStruct(int M, int N, int K, int nc, int kc, int padding_size)
            : M(M), N(N), K(K), nc(nc), kc(kc), padding_size(padding_size){{}}
    }};
    static std::list<SimpleStruct> params_list;
    static void CreateList()
    {{
"""
        with open(input_file_path, 'r') as load_f:
            for line in load_f:
                load_dict = json.loads(line)
                MKN = load_dict["input"]
                cfg = load_dict["config"]["entity"]
                cc_code+=f"""        params_list.push_back(SimpleStruct({MKN[2][0]}, {MKN[2][2]}, {MKN[2][1]}, {cfg[1][-1][-1]}, {cfg[2][-1][-1]}, {cfg[6][-1]}));\n"""
        cc_code += f"""    }}
}};
#endif"""

        f = open(output_file_path, 'w')
        f.write(cc_code)
        f.close()


if __name__ == "__main__":
    main()
