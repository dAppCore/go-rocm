// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"context"

	rocmmodel "dappco.re/go/rocm/model"
)

// ROCmCacheProfileReporter exposes the live runtime cache profile used by
// reactive model-route consumers.
type ROCmCacheProfileReporter interface {
	CacheProfile(context.Context) (rocmmodel.CacheProfile, error)
}
