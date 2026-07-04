// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && rocm_legacy_server

package rocm

import (
	"context"

	rocmmodel "dappco.re/go/rocm/model"
)

func (m *rocmModel) CacheProfile(ctx context.Context) (profile rocmmodel.CacheProfile, err error) {
	if m != nil {
		m.clearLastError()
	}
	defer func() {
		if m != nil && err != nil {
			m.setLastFailure(err)
		}
	}()
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return rocmmodel.CacheProfile{}, err
		}
	}
	return rocmmodel.CacheProfile{}, nil
}
