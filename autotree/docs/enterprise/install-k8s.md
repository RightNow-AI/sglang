# Kubernetes install

This is the shortest fresh-cluster path for the current CPU `autotree-serve`
image. With a Kubernetes cluster, ingress/DNS if desired, and an image already
available to the cluster, the install and smoke test should fit inside 30
minutes. Image compilation and first model download are environment-dependent
and are not included in that timing claim; this repository does not publish a
prebuilt image.

## 1. Make the image available

Build from the repository root and push it to a registry your nodes can pull:

```bash
docker build -t registry.example/autotree-serve:0.1.0 .
docker push registry.example/autotree-serve:0.1.0
```

For a local Kind cluster, replace the push with:

```bash
kind load docker-image registry.example/autotree-serve:0.1.0
```

The root Dockerfile installs the CPU runtime. `values-gpu.yaml` is a scheduling
overlay for a separately built and validated GPU image; applying it to the root
image does not make the server GPU-capable.

This repository currently ships Helm resources and an HPA, not a Kubernetes
operator or custom resource controller. Operator-managed rollouts remain
roadmap work.

## 2. Install metrics and the custom-metric adapter

The chart exposes `/metrics` through Service scrape annotations. Its HPA uses
the real Prometheus gauge `active_branches` as a per-pod KV-pressure proxy:
more live branches imply more simultaneously retained tree KV state. The server
does not currently export allocated-KV-pages divided by capacity, so the chart
does not claim that `active_branches` is a direct memory percentage.

If the cluster already has Prometheus and Prometheus Adapter, add this custom
rule to the adapter and skip the chart installs below:

```yaml
rules:
  custom:
    - seriesQuery: 'active_branches{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: namespace}
          pod: {resource: pod}
      name:
        matches: '^active_branches$'
        as: active_branches
      metricsQuery: 'max(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
```

For a new cluster, one working stack is:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set grafana.sidecar.dashboards.enabled=true

cat >/tmp/autotree-adapter-values.yaml <<'YAML'
prometheus:
  url: http://monitoring-kube-prometheus-prometheus.monitoring.svc
  port: 9090
rules:
  default: false
  custom:
    - seriesQuery: 'active_branches{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: namespace}
          pod: {resource: pod}
      name:
        matches: '^active_branches$'
        as: active_branches
      metricsQuery: 'max(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
YAML

helm upgrade --install prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace monitoring -f /tmp/autotree-adapter-values.yaml
```

## 3. Install AutoTree

```bash
helm upgrade --install autotree deploy/helm/autotree \
  --namespace autotree --create-namespace \
  --set image.repository=registry.example/autotree-serve \
  --set image.tag=0.1.0 \
  --set serviceMonitor.enabled=true \
  --set serviceMonitor.labels.release=monitoring \
  --wait --timeout 10m
```

Defaults run the real GPT-2 TreeKV CPU path with no authentication, preserving
the frictionless quickstart. Production authentication is opt-in; see
[security.md](security.md).

Check the workload, HPA, and custom metric:

```bash
kubectl -n autotree get pods,svc,hpa,pdb
kubectl get --raw \
  '/apis/custom.metrics.k8s.io/v1beta1/namespaces/autotree/pods/*/active_branches'
```

Smoke test:

```bash
kubectl -n autotree port-forward service/autotree-autotree 8000:80
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/metrics
```

## 4. Load the Grafana dashboard

The kube-prometheus-stack dashboard sidecar imports labeled ConfigMaps:

```bash
kubectl -n monitoring create configmap autotree-dashboard \
  --from-file=autotree-overview.json=grafana/autotree-overview.json \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n monitoring label configmap autotree-dashboard grafana_dashboard=1 --overwrite
```

The dashboard uses only metrics emitted by `autotree-serve`: `kv_reuse_ratio`,
`useful_token_ratio`, `active_branches`, `branch_events_total`,
`tokens_per_second`, `ttft_seconds`, `requests_total`,
`capacity_rejections_total`, and `quota_rejections_total`.

## Plain manifests

Non-Helm consumers can apply `deploy/k8s/autotree.yaml`. It is generated from
the chart by `deploy/render.py`; replace the `autotree:local` image before
applying it to a remote cluster. Helm remains required only for regenerating or
checking that file, not for applying it.

The plain manifest uses Prometheus scrape annotations and does not include the
optional Prometheus-Operator `ServiceMonitor`; configure the cluster's existing
Prometheus discovery accordingly.
