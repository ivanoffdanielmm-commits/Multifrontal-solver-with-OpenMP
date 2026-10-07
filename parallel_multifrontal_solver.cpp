#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <functional>
#include <omp.h>

using namespace std;
using DenseMatrix = vector<vector<double>>;

struct ContributionBlock {
    vector<int> rows, cols;
    DenseMatrix values;
};

struct FrontalMatrix {
    int node;
    vector<int> globalRows;
    DenseMatrix F;
    vector<int> fullySummed;
    vector<double> Ddiag;
    vector<tuple<int,int,DenseMatrix>> D2x2_inv;
    DenseMatrix L_perm;
    vector<int> perm;
    vector<int> invPerm;
};

vector<FrontalMatrix> frontalMatrices;
vector<ContributionBlock> contribBlocks;
vector<char> has_contrib;

struct AssemblyTree {
    int n;
    vector<int> parent;
    vector<vector<int>> children;
    vector<vector<int>> nodeVars;
    vector<vector<int>> fullySummedVars;
};

vector<tuple<int,int,double>> readMTX(const string& filename, int &N) {
    ifstream file(filename);
    if (!file.is_open()) { cerr << "Ошибка открытия файла " << filename << "\n"; exit(1); }
    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;
        break;
    }
    stringstream ss(line);
    int M, NNZ;
    ss >> N >> M >> NNZ;
    vector<tuple<int,int,double>> A;
    for (int i = 0; i < NNZ; ++i) {
        int u, v; double val;
        file >> u >> v >> val;
        u--; v--;
        if (u < v) swap(u, v);
        A.push_back({u, v, val});
    }
    return A;
}

// Minimum Degree Ordering
vector<int> minimumDegreeOrdering(int N, const vector<tuple<int,int,double>>& A) {
    vector<set<int>> adj(N);
    for (const auto& e : A) {
        int u = get<0>(e), v = get<1>(e);
        if (u != v) { adj[u].insert(v); adj[v].insert(u); }
    }
    vector<int> perm;
    vector<bool> active(N, true);
    for (int step = 0; step < N; ++step) {
        int best = -1, min_deg = 1e9;
        for (int i = 0; i < N; ++i) {
            if (active[i] && adj[i].size() < min_deg) {
                min_deg = adj[i].size(); best = i;
            }
        }
        perm.push_back(best);
        active[best] = false;
        vector<int> neigh;
        for (int v : adj[best]) if (active[v]) neigh.push_back(v);
        for (size_t i = 0; i < neigh.size(); ++i) {
            for (size_t j = i+1; j < neigh.size(); ++j) {
                int u = neigh[i], v = neigh[j];
                adj[u].insert(v); adj[v].insert(u);
            }
        }
        for (int v : neigh) adj[v].erase(best);
    }
    return perm;
}

//Дерево исключения
AssemblyTree buildEliminationTree(int N, const vector<tuple<int,int,double>>& permA) {
    AssemblyTree tree; tree.n = N;
    tree.parent.assign(N, -1);
    tree.children.resize(N);
    tree.nodeVars.resize(N);
    vector<int> ancestor(N, -1);
    vector<vector<int>> lowerAdj(N);
    for (const auto& e : permA) {
        int u = get<0>(e), v = get<1>(e);
        if (u > v) lowerAdj[u].push_back(v);
    }
    for (int i = 0; i < N; ++i) {
        for (int j : lowerAdj[i]) {
            int j_root = j;
            while (ancestor[j_root] != -1 && ancestor[j_root] != i) {
                int l = ancestor[j_root];
                ancestor[j_root] = i;
                j_root = l;
            }
            if (ancestor[j_root] == -1) {
                ancestor[j_root] = i;
                tree.parent[j_root] = i;
            }
        }
    }
    for (int i = 0; i < N; ++i)
        if (tree.parent[i] != -1) tree.children[tree.parent[i]].push_back(i);
    for (int i = 0; i < N; ++i) tree.nodeVars[i].push_back(i);
    for (const auto& e : permA) {
        int u = get<0>(e), v = get<1>(e);
        if (u > v) tree.nodeVars[v].push_back(u);
    }
    return tree;
}

void computeFullySummedVars(AssemblyTree &tree) {
    int n = tree.n;
    set<int> allVars;
    for (auto &nv : tree.nodeVars) for (int v : nv) allVars.insert(v);
    map<int,int> lca;
    for (int v : allVars) {
        vector<int> nodes;
        for (int i = 0; i < n; ++i)
            if (find(tree.nodeVars[i].begin(), tree.nodeVars[i].end(), v) != tree.nodeVars[i].end())
                nodes.push_back(i);
        int cur = nodes[0];
        for (size_t k = 1; k < nodes.size(); ++k) {
            int a = cur, b = nodes[k];
            set<int> path;
            while (a != -1) { path.insert(a); a = tree.parent[a]; }
            while (b != -1) { if (path.count(b)) { cur = b; break; } b = tree.parent[b]; }
        }
        lca[v] = cur;
    }
    tree.fullySummedVars.assign(n, vector<int>());
    for (auto &[v,node] : lca) if (node != -1) tree.fullySummedVars[node].push_back(v);
    for (auto &fs : tree.fullySummedVars) sort(fs.begin(), fs.end());
}

void extendAdd(DenseMatrix &F, const vector<int> &Frows, const ContributionBlock &C) {
    map<int,int> local;
    for (size_t i = 0; i < Frows.size(); ++i) local[Frows[i]] = i;
    for (size_t i = 0; i < C.rows.size(); ++i) {
        int gi = C.rows[i], li = local[gi];
        for (size_t j = 0; j < C.cols.size(); ++j) {
            int gj = C.cols[j], lj = local[gj];
            F[li][lj] += C.values[i][j];
        }
    }
}

DenseMatrix permuteMatrix(const DenseMatrix &A, const vector<int> &perm) {
    int n = A.size();
    DenseMatrix B(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            B[i][j] = A[perm[i]][perm[j]];
    return B;
}

//Факторизация фронта
bool factorizeFrontal(FrontalMatrix &front, double pivotTol = 0.1) {
    int n = front.F.size();
    vector<int> fs = front.fullySummed;
    vector<int> nonFs;
    for (int i = 0; i < n; ++i)
        if (find(fs.begin(), fs.end(), i) == fs.end()) nonFs.push_back(i);
    sort(fs.begin(), fs.end());
    sort(nonFs.begin(), nonFs.end());

    vector<int> perm;
    perm.insert(perm.end(), fs.begin(), fs.end());
    perm.insert(perm.end(), nonFs.begin(), nonFs.end());
    front.perm = perm;

    DenseMatrix F_perm = permuteMatrix(front.F, perm);
    DenseMatrix L_perm(n, vector<double>(n, 0.0));

    int k = fs.size();
    int step = 0;

    while (step < k) {
        int best = -1;
        double bestVal = 0.0;
        for (int idx = step; idx < k; ++idx) {
            double piv = F_perm[idx][idx];
            if (fabs(piv) > pivotTol) {
                double colMax = 0.0;
                for (int i = step; i < n; ++i) if (i != idx) colMax = max(colMax, fabs(F_perm[i][idx]));
                if (fabs(piv) >= 0.1 * colMax) {
                    if (fabs(piv) > bestVal) { bestVal = fabs(piv); best = idx; }
                }
            }
        }

        if (best != -1) {
            int p = best;
            if (p != step) {
                for (int i = 0; i < n; ++i) swap(F_perm[step][i], F_perm[p][i]);
                for (int i = 0; i < n; ++i) swap(F_perm[i][step], F_perm[i][p]);
                for (int c = 0; c < step; ++c) swap(L_perm[step][c], L_perm[p][c]);
                swap(front.perm[step], front.perm[p]);
            }
            double dp = F_perm[step][step];
            front.Ddiag.push_back(dp);

            vector<double> col_j(n, 0.0);
            for (int j = step + 1; j < n; ++j) col_j[j] = F_perm[j][step] / dp;

            #pragma omp parallel for schedule(static) if((n - step) > 128)
            for (int i = step + 1; i < n; ++i) {
                double mult_i = F_perm[i][step] / dp;
                L_perm[i][step] = mult_i;
                for (int j = step + 1; j < n; ++j) {
                    F_perm[i][j] -= mult_i * dp * col_j[j];
                }
            }
            step++;
            continue;
        }

        bool found2x2 = false;
        for (int i = step; i < k && !found2x2; ++i) {
            for (int j = i + 1; j < k; ++j) {
                double a = F_perm[i][i], b = F_perm[i][j], c = F_perm[j][j];
                double det = a * c - b * b;
                if (fabs(det) > 1e-12) {
                    if (i != step) {
                        for (int x = 0; x < n; ++x) swap(F_perm[step][x], F_perm[i][x]);
                        for (int x = 0; x < n; ++x) swap(F_perm[x][step], F_perm[x][i]);
                        for (int c_idx = 0; c_idx < step; ++c_idx) swap(L_perm[step][c_idx], L_perm[i][c_idx]);
                        swap(front.perm[step], front.perm[i]);
                        if (j == step) j = i;
                    }
                    if (j != step + 1) {
                        for (int x = 0; x < n; ++x) swap(F_perm[step+1][x], F_perm[j][x]);
                        for (int x = 0; x < n; ++x) swap(F_perm[x][step+1], F_perm[x][j]);
                        for (int c_idx = 0; c_idx < step; ++c_idx) swap(L_perm[step+1][c_idx], L_perm[j][c_idx]);
                        swap(front.perm[step+1], front.perm[j]);
                    }
                    a = F_perm[step][step]; b = F_perm[step][step+1]; c = F_perm[step+1][step+1];
                    front.Ddiag.push_back(1.0); front.Ddiag.push_back(1.0);

                    double invDet = 1.0 / det;
                    DenseMatrix invBlock = {{c * invDet, -b * invDet}, {-b * invDet, a * invDet}};
                    front.D2x2_inv.push_back({step, step+1, invBlock});

                    vector<double> w0_c_vec(n, 0.0), w1_c_vec(n, 0.0);
                    for (int c_idx = step + 2; c_idx < n; ++c_idx) {
                        w0_c_vec[c_idx] = F_perm[c_idx][step];
                        w1_c_vec[c_idx] = F_perm[c_idx][step+1];
                    }

                    #pragma omp parallel for schedule(static) if((n - step) > 128)
                    for (int r = step + 2; r < n; ++r) {
                        double w0 = F_perm[r][step], w1 = F_perm[r][step+1];
                        double mult0 = w0 * invBlock[0][0] + w1 * invBlock[1][0];
                        double mult1 = w0 * invBlock[0][1] + w1 * invBlock[1][1];
                        L_perm[r][step] = mult0; L_perm[r][step+1] = mult1;
                        for (int c_idx = step + 2; c_idx < n; ++c_idx) {
                            F_perm[r][c_idx] -= (mult0 * w0_c_vec[c_idx] + mult1 * w1_c_vec[c_idx]);
                        }
                    }
                    step += 2; found2x2 = true; break;
                }
            }
        }

        if (!found2x2) {
            double piv = F_perm[step][step];
            if (fabs(piv) < 1e-12) {
                piv = (piv < 0.0) ? -1e-12 : 1e-12;
                F_perm[step][step] = piv;
            }
            double dp = piv;
            front.Ddiag.push_back(dp);

            vector<double> col_j_fb(n, 0.0);
            for (int j = step + 1; j < n; ++j) col_j_fb[j] = F_perm[j][step] / dp;

            #pragma omp parallel for schedule(static) if((n - step) > 128)
            for (int i = step + 1; i < n; ++i) {
                double mult_i = F_perm[i][step] / dp;
                L_perm[i][step] = mult_i;
                for (int j = step + 1; j < n; ++j) {
                    F_perm[i][j] -= mult_i * dp * col_j_fb[j];
                }
            }
            step++;
        }
    }

    front.L_perm = L_perm;
    front.invPerm.resize(n);
    for (int i = 0; i < n; ++i) front.invPerm[front.perm[i]] = i;
    front.F = permuteMatrix(F_perm, front.invPerm);
    return true;
}

//Обработка узла дерева
void factorizeNode(int node, const AssemblyTree &tree, const vector<tuple<int,int,double>> &node_edges) {
    set<int> globSet(tree.nodeVars[node].begin(), tree.nodeVars[node].end());
    for (int child : tree.children[node]) {
        if (has_contrib[child]) {
            for (int g : contribBlocks[child].rows) globSet.insert(g);
        }
    }

    vector<int> globalRows(globSet.begin(), globSet.end());
    int n = globalRows.size();

    if (n > 5000) {
        #pragma omp critical
        {
            double mb = (double)n * n * sizeof(double) / (1024.0 * 1024.0);
            cout << "  [RAM Guard] Корень дерева. Узел " << node << ": матрица " << n << "x" << n
                 << " (~" << fixed << setprecision(1) << mb << " МБ памяти)\n";
        }
    }

    FrontalMatrix front;
    front.node = node;
    front.globalRows = globalRows;
    front.F.assign(n, vector<double>(n, 0.0));

    for (const auto& edge : node_edges) {
        int i = get<0>(edge), j = get<1>(edge);
        double val = get<2>(edge);
        auto it_i = find(globalRows.begin(), globalRows.end(), i);
        auto it_j = find(globalRows.begin(), globalRows.end(), j);
        if (it_i != globalRows.end() && it_j != globalRows.end()) {
            int li = distance(globalRows.begin(), it_i);
            int lj = distance(globalRows.begin(), it_j);
            front.F[li][lj] += val;
            if (li != lj) front.F[lj][li] += val;
        }
    }

    for (int child : tree.children[node]) {
        if (has_contrib[child]) {
            extendAdd(front.F, globalRows, contribBlocks[child]);
            // Очистка RAM для предотвращения OOM
            contribBlocks[child].rows.clear();
            contribBlocks[child].rows.shrink_to_fit();
            contribBlocks[child].cols.clear();
            contribBlocks[child].cols.shrink_to_fit();
            contribBlocks[child].values.clear();
            contribBlocks[child].values.shrink_to_fit();
        }
    }

    vector<int> fullySummedLocal;
    for (int g : tree.fullySummedVars[node]) {
        auto it = find(globalRows.begin(), globalRows.end(), g);
        if (it != globalRows.end()) fullySummedLocal.push_back(distance(globalRows.begin(), it));
    }
    front.fullySummed = fullySummedLocal;

    factorizeFrontal(front, 1e-12);

    set<int> notFullySummed;
    for (int g : globalRows) {
        if (find(tree.fullySummedVars[node].begin(), tree.fullySummedVars[node].end(), g) == tree.fullySummedVars[node].end())
            notFullySummed.insert(g);
    }
    if (!notFullySummed.empty()) {
        vector<int> nfsVec(notFullySummed.begin(), notFullySummed.end());
        int sz = nfsVec.size();
        ContributionBlock cb;
        cb.rows = cb.cols = nfsVec;
        cb.values.assign(sz, vector<double>(sz, 0.0));
        for (int i = 0; i < sz; ++i) {
            int gi = nfsVec[i];
            int li = distance(globalRows.begin(), find(globalRows.begin(), globalRows.end(), gi));
            for (int j = 0; j < sz; ++j) {
                int gj = nfsVec[j];
                int lj = distance(globalRows.begin(), find(globalRows.begin(), globalRows.end(), gj));
                cb.values[i][j] = front.F[li][lj];
            }
        }
        contribBlocks[node] = cb;
        has_contrib[node] = 1;
    }

    frontalMatrices[node] = front;
}

//Решение системы
vector<double> solveSystem(const vector<double> &b, const vector<int>& order) {
    vector<double> x = b;
    for (int node : order) {
        auto &front = frontalMatrices[node];
        if (front.globalRows.empty()) continue;
        int nloc = front.globalRows.size();
        vector<double> y_perm(nloc);
        for (int i = 0; i < nloc; ++i) {
            int oldIdx = front.perm[i];
            y_perm[i] = x[front.globalRows[oldIdx]];
        }
        for (int j = 0; j < nloc; ++j)
            for (int i = j+1; i < nloc; ++i)
                y_perm[i] -= front.L_perm[i][j] * y_perm[j];
        for (int i = 0; i < nloc; ++i) {
            int oldIdx = front.perm[i];
            x[front.globalRows[oldIdx]] = y_perm[i];
        }
    }

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        auto &front = frontalMatrices[*it];
        if (front.globalRows.empty()) continue;
        int nloc = front.globalRows.size();
        vector<double> y_perm(nloc);
        for (int i = 0; i < nloc; ++i) {
            int oldIdx = front.perm[i];
            y_perm[i] = x[front.globalRows[oldIdx]];
        }
        int k = front.fullySummed.size();
        for (int idx = 0; idx < k; ++idx) y_perm[idx] /= front.Ddiag[idx];
        for (const auto& blk : front.D2x2_inv) {
            int i1 = get<0>(blk), i2 = get<1>(blk);
            const DenseMatrix &inv = get<2>(blk);
            double v1 = y_perm[i1], v2 = y_perm[i2];
            y_perm[i1] = inv[0][0]*v1 + inv[0][1]*v2;
            y_perm[i2] = inv[1][0]*v1 + inv[1][1]*v2;
        }
        for (int j = nloc-1; j >= 0; --j)
            for (int i = j-1; i >= 0; --i)
                y_perm[i] -= front.L_perm[j][i] * y_perm[j];
        for (int i = 0; i < nloc; ++i) {
            int oldIdx = front.perm[i];
            x[front.globalRows[oldIdx]] = y_perm[i];
        }
    }
    return x;
}

//MAIN
int main(int argc, char* argv[]) {
    string filename = (argc > 1) ? argv[1] : "494_bus.mtx";
    cout << "=== Параллельный Мультифронтальный Решатель СЛАУ (OpenMP) ===\n";
    cout << "Потоков OpenMP доступно: " << omp_get_max_threads() << "\n\n";

    double t_start = omp_get_wtime();
    int N;
    auto A_raw = readMTX(filename, N);
    cout << "Матрица загружена. N = " << N << ", NNZ = " << A_raw.size() << endl;

    double t_mdo = omp_get_wtime();
    vector<int> perm = minimumDegreeOrdering(N, A_raw);
    vector<int> invPerm(N);
    for (int i = 0; i < N; ++i) invPerm[perm[i]] = i;
    cout << "ШАГ 1. Minimum Degree Ordering. Время: " << fixed << setprecision(2) << omp_get_wtime() - t_mdo << " сек\n";

    double t_sym = omp_get_wtime();
    vector<tuple<int,int,double>> A_perm;
    for (const auto& e : A_raw) {
        int u = invPerm[get<0>(e)], v = invPerm[get<1>(e)];
        if (u < v) swap(u, v);
        A_perm.push_back({u, v, get<2>(e)});
    }
    AssemblyTree tree = buildEliminationTree(N, A_perm);
    computeFullySummedVars(tree);

    vector<int> order;
    std::function<void(int)> dfs = [&](int node) {
        for (int child : tree.children[node]) dfs(child);
        order.push_back(node);
    };
    for (int i = 0; i < N; ++i) if (tree.parent[i] == -1) dfs(i);

    frontalMatrices.resize(N);
    contribBlocks.resize(N);
    has_contrib.assign(N, 0);

    vector<vector<tuple<int,int,double>>> edges_per_node(N);
    for (const auto& e : A_perm) {
        int u = get<0>(e), v = get<1>(e);
        edges_per_node[min(u,v)].push_back(e);
    }

    vector<int> level(N, 0);
    int max_level = 0;
    for (int node : order) {
        int l = 0;
        for (int child : tree.children[node]) l = max(l, level[child] + 1);
        level[node] = l;
        max_level = max(max_level, l);
    }
    vector<vector<int>> levels(max_level + 1);
    for (int node : order) levels[level[node]].push_back(node);

    cout << "ШАГ 2. Символьный анализ. Уровней в дереве: " << max_level+1 << ". Время: " << omp_get_wtime() - t_sym << " сек\n";

    // Сбор статистики для диплома
    int max_parallel_nodes = 0;
    for (int l = 0; l <= max_level; ++l) {
        max_parallel_nodes = max(max_parallel_nodes, (int)levels[l].size());
    }
    cout << "  Пиковая ширина дерева (макс. параллельных задач): " << max_parallel_nodes << "\n";

    double t_fact = omp_get_wtime();
    for (int l = 0; l <= max_level; ++l) {
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < levels[l].size(); ++i) {
            int node = levels[l][i];
            factorizeNode(node, tree, edges_per_node[node]);
        }
    }
    cout << "ШАГ 3. Параллельная LDL^T факторизация. Время: " << omp_get_wtime() - t_fact << " сек\n";

    double t_solve = omp_get_wtime();
    vector<double> b(N, 0.0);
    for (const auto &e : A_raw) {
        int i = get<0>(e), j = get<1>(e);
        double val = get<2>(e);
        b[i] += val;
        if (i != j) b[j] += val;
    }
    vector<double> b_perm(N);
    for (int i = 0; i < N; ++i) b_perm[invPerm[i]] = b[i];

    vector<double> x_perm = solveSystem(b_perm, order);

    vector<double> x(N);
    for (int i = 0; i < N; ++i) x[perm[i]] = x_perm[i];

    double maxerr = 0.0;
    for (int i = 0; i < N; ++i) maxerr = max(maxerr, fabs(x[i] - 1.0));
    cout << "ШАГ 4. Решение СЛАУ. Время: " << omp_get_wtime() - t_solve << " сек\n";
    cout << "-------------------------------------------\n";
    cout << "Общее время: " << omp_get_wtime() - t_start << " сек\n";
    cout << "Максимальная ошибка ||x_calc - 1.0||: " << maxerr << endl;
    return 0;
}
