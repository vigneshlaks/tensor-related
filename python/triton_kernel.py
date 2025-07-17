


def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, num_stages):
    # 
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0) 

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, )
        


