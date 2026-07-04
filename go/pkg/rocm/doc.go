// SPDX-Licence-Identifier: EUPL-1.2

// Package rocm exposes the sparse, embeddable ROCm runtime contract.
//
// The implementation currently forwards to the module-root runtime package so
// existing callers keep working while go-rocm grows a package layout closer to
// go-mlx's pkg/model + pkg/native split. New application integrations should
// prefer this package path for model-reactive routing, production defaults, and
// multi-lane artifact negotiation.
package rocm
