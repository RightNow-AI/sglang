{{- define "autotree.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "autotree.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name (include "autotree.name" .) | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "autotree.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | quote }}
app.kubernetes.io/name: {{ include "autotree.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "autotree.selectorLabels" -}}
app.kubernetes.io/name: {{ include "autotree.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
