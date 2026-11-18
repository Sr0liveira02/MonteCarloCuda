function write_sparse_to_hdf5(A, outFile)
    [n_rows, n_cols] = size(A);
    nnzA = nnz(A);

    % --- Delete file if exists ---
    if isfile(outFile)
        delete(outFile);
    end

    % --- Pre-create HDF5 datasets ---
    h5create(outFile, '/CSR/row_ptr', n_rows+1,  'Datatype', 'int32');
    h5create(outFile, '/CSR/col_ind', nnzA,       'Datatype', 'int32');
    h5create(outFile, '/CSR/values',  nnzA,       'Datatype', 'single');

    % --- Build CSR row-by-row (never hold full COO in memory) ---
    % MATLAB sparse is stored column-major internally, but we can still
    % extract one row at a time cheaply via logical indexing.
    %
    % For very large matrices, use the non-zero col/val per row directly.

    row_ptr = zeros(n_rows+1, 1, 'int32');   % small: n_rows+1 ints
    col_buf = zeros(nnzA, 1, 'int32');        % full col_ind buffer
    val_buf = zeros(nnzA, 1, 'single');       % full values buffer

    ptr = int32(0);
    for r = 1:n_rows
        row_ptr(r) = ptr;
        % Extract one row — MATLAB only materialises this row's data
        rowVec = A(r, :);
        [~, cols, vals] = find(rowVec);       % cols/vals for this row only
        nz = int32(numel(vals));
        if nz > 0
            range = ptr+1 : ptr+nz;
            col_buf(range) = int32(cols) - int32(1);  % 0-based
            val_buf(range) = single(vals);
            ptr = ptr + nz;
        end
    end
    row_ptr(n_rows+1) = ptr;

    % --- Write in one shot (already built in-place, no extra copies) ---
    h5write(outFile, '/CSR/row_ptr', row_ptr);
    h5write(outFile, '/CSR/col_ind', col_buf);
    h5write(outFile, '/CSR/values',  val_buf);

    h5writeatt(outFile, '/CSR', 'n_rows', int32(n_rows));
    h5writeatt(outFile, '/CSR', 'n_cols', int32(n_cols));
    fprintf('HDF5 written: %s\n', outFile);
end

function convert_suitesparse_to_hdf5(matFile, outFile)

    S = load(matFile);

    % --- Case 1: SuiteSparse format ---
    if isfield(S, 'Problem') && isfield(S.Problem, 'A')
        A = S.Problem.A;

    % --- Case 2: plain A in file ---
    elseif isfield(S, 'A')
        A = S.A;

    % --- Case 3: try to find any sparse matrix ---
    else
        vars = fieldnames(S);
        found = false;

        for i = 1:numel(vars)
            candidate = S.(vars{i});
            if issparse(candidate)
                A = candidate;
                found = true;
                fprintf('Found sparse matrix in variable "%s"\n', vars{i});
                break;
            end
        end

        if ~found
            error('No sparse matrix found in MAT file.');
        end
    end

    % --- Safety check ---
    if ~issparse(A)
        error('Matrix A is not sparse.');
    end

    fprintf('Matrix size: %d x %d, nnz = %d\n', size(A,1), size(A,2), nnz(A));

    % --- Write to your CSR HDF5 ---
    write_sparse_to_hdf5(A, outFile);
end

convert_suitesparse_to_hdf5('Spielman_k600.mat', 'Spielman_k600.h5');