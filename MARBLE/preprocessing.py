

import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit

from MARBLE import geometry as g
from MARBLE import utils


def construct_dataset(
    anchor,
    vector,
    label=None,
    mask=None,
    graph_type="cknn",
    k=20,
    delta=1.0,
    frac_geodesic_nb=1.5,
    spacing=0.0,
    number_of_resamples=1,
    var_explained=0.9,
    local_gauges=False,
    seed=None,
    metric="euclidean",
    number_of_eigenvectors=None,
):
    """Construct PyG dataset from node positions and features.

    Args:
        anchor: matrix with positions of points
        vector: matrix with feature values for each point
        label: any additional data labels used for plotting only
        mask: boolean array, that will be forced to be close (default is None)
        graph_type: type of nearest-neighbours graph: cknn (default), knn or radius
        k: number of nearest-neighbours to construct the graph
        delta: argument for cknn graph construction to decide the radius for each point
        frac_geodesic_nb: number of geodesic neighbours to fit the gauges to
                          to map to tangent space k*frac_geodesic_nb
        spacing: stopping criterion for furthest point sampling
        number_of_resamples: number of furthest point sampling runs to prevent bias
        var_explained: fraction of variance explained by the local gauges
        local_gauges: if True, it will try to compute local gauges if it can
        seed: Specify for reproducibility in the furthest point sampling.
        metric: metric used to fit proximity graph
        number_of_eigenvectors: int number of eigenvectors to use. Default: None, meaning use all.
    """

    anchor = [torch.tensor(a).float() for a in utils.to_list(anchor)]
    vector = [torch.tensor(v).float() for v in utils.to_list(vector)]
    num_node_features = vector[0].shape[1]

    if label is None:
        label = [torch.arange(len(a)) for a in utils.to_list(anchor)]
    else:
        label = [torch.tensor(lab).float() for lab in utils.to_list(label)]

    if mask is None:
        mask = [torch.zeros(len(a), dtype=torch.bool) for a in utils.to_list(anchor)]
    else:
        mask = [torch.tensor(m) for m in utils.to_list(mask)]

    if spacing == 0.0:
        number_of_resamples = 1

    data_list = []
    for i, (a, v, l, m) in enumerate(zip(anchor, vector, label, mask)):
        for _ in range(number_of_resamples):
            if len(a) != 0:
                # even sampling of points
                if seed is None:
                    start_idx = torch.randint(low=0, high=len(a), size=(1,))
                else:
                    start_idx = 0

                sample_ind, _ = g.furthest_point_sampling(a, spacing=spacing, start_idx=start_idx)
                sample_ind, _ = torch.sort(sample_ind)  # this will make postprocessing easier
                a_, v_, l_, m_ = (
                    a[sample_ind],
                    v[sample_ind],
                    l[sample_ind],
                    m[sample_ind],
                )

                # fit graph to point cloud
                edge_index, edge_weight = g.fit_graph(
                    a_, graph_type=graph_type, par=k, delta=delta, metric=metric
                )

                # define data object
                data_ = Data(
                    pos=a_,
                    x=v_,
                    label=l_,
                    mask=m_,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    num_nodes=len(a_),
                    num_node_features=num_node_features,
                    y=torch.ones(len(a_), dtype=int) * i,
                    sample_ind=sample_ind,
                )

                data_list.append(data_)

    # collate datasets
    batch = Batch.from_data_list(data_list)
    batch.degree = k
    batch.number_of_resamples = number_of_resamples

    # split into training/validation/test datasets
    split = RandomNodeSplit(split="train_rest", num_val=0.1, num_test=0.1)
    split(batch)

    return _compute_geometric_objects(
        batch,
        local_gauges=local_gauges,
        n_geodesic_nb=k * frac_geodesic_nb,
        var_explained=var_explained,
        number_of_eigenvectors=number_of_eigenvectors,
    )


def _compute_geometric_objects(
    data,
    n_geodesic_nb=10,
    var_explained=0.9,
    local_gauges=False,
    number_of_eigenvectors=None,
):
    """
    Compute geometric objects used later: local gauges,
    gradient kernels, and scalar Laplacian spectrum.

    In the simplified version:
    - we do NOT compute the connection Laplacian `Lc`
    - only scalar Laplacian eigen-decomposition is used.
    """

    n, dim_emb = data.pos.shape
    dim_signal = data.x.shape[1]
    print(f"\n---- Embedding dimension: {dim_emb}", end="")
    print(f"\n---- Signal dimension: {dim_signal}", end="")

    # disable vector manifold computations in simple cases
    if dim_signal == 1:
        print("\nSignal dimension is 1, so manifold computations are disabled!")
        local_gauges = False
    if dim_emb <= 2:
        print("\nEmbedding dimension <= 2, so manifold computations are disabled!")
        local_gauges = False
    if dim_emb != dim_signal:
        print("\nEmbedding dimension /= signal dimension, so manifold computations are disabled!")

    # gauges: either local (if enabled) or global identity frames
    if local_gauges:
        try:
            gauges, Sigma = g.compute_gauges(data, n_geodesic_nb=n_geodesic_nb)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise Exception(
                "\nCould not compute gauges (possibly data is too sparse or the "
                "number of neighbours is too small)"
            ) from exc
    else:
        gauges = torch.eye(dim_emb).repeat(n, 1, 1)
        Sigma = None

    # scalar Laplacian
    L = g.compute_laplacian(data)

    # directional kernels and (optionally) manifold dimension
    if local_gauges and Sigma is not None:
        data.dim_man = g.manifold_dimension(Sigma, frac_explained=var_explained)
        print(f"---- Manifold dimension: {data.dim_man}")

        gauges = gauges[:, :, : data.dim_man]

        print("\n---- Computing kernels ... ", end="")
        kernels = g.gradient_op(data.pos, data.edge_index, gauges)
        # in the simplified version we keep kernels as-is (no connection Laplacian)
    else:
        print("\n---- Computing kernels ... ", end="")
        kernels = g.gradient_op(data.pos, data.edge_index, gauges)
        data.dim_man = None

    # Laplacian spectrum
    if number_of_eigenvectors is None:
        print(
            """\n---- Computing full spectrum ...
              (if this takes too long, then run construct_dataset()
              with number_of_eigenvectors specified) """,
            end="",
        )
    else:
        print(
            f"\n---- Computing spectrum with {number_of_eigenvectors} eigenvectors...",
            end="",
        )

    L = g.compute_eigendecomposition(L, k=number_of_eigenvectors)

    # store results in data object
    data.kernels = [
        utils.to_SparseTensor(K.coalesce().indices(), value=K.coalesce().values())
        for K in kernels
    ]
    data.L = L
    data.gauges = gauges
    data.local_gauges = local_gauges
    data.Lc = None  # explicit: no connection Laplacian in simplified version

    return data
