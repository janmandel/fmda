
import numpy as np


class GridMoistureModel:

    Tk = np.array([1, 10, 100, 1000]) * 3600.0    # nominal fuel delays
    r0 = 0.05                                     # threshold rainfall [mm/h]
    rk = 8                                        # saturation rain intensity [mm/h]
    Trk = 14 * 3600                               # time constant for wetting model [s]
    S = 2.5                                       # saturation intensity [dimensionless]


    def __init__(self, m0 = None, Tk = None, P0 = None):
        """
        Initialize the model with given position and moisture levels.

          m0  - the initial condition for the entire grid (shape: grid0 x grid1 x num_fuels)
          Tk  - drying/wetting time constants of simulated fuels (one per fuel), default [1 10 100 1000]
          P0  - initial state error covariance

        """
        s0,s1,k = m0.shape
        if Tk is not None:
            self.Tk = Tk
        dim = k + 2
        assert k == len(self.Tk)
        self.m_ext = np.zeros((s0,s1,dim))
        if m0 is not None:
            self.m_ext[:,:,:k] = m0

        # note: the moisture advance will proceed by fuel moisture types
        # thus we only need space for one class at a time
        self.m_i = np.zeros((s0,s1))
        self.mn_i = np.zeros((s0,s1))
        self.rlag = np.zeros((s0,s1))
        self.equi = np.zeros((s0,s1))
        self.model_ids = np.zeros((s0,s1))

        # state covariance current and forecasted
        self.P = np.zeros((s0,s1,dim,dim))
        self.P2 = np.zeros((dim,dim))
        for s in np.ndindex((s0,s1)):
          self.P[s[0],s[1],:,:] = P0

        # fill out the fixed parts of the jacobian
        self.J = np.zeros((s0,s1,dim,dim))
        self.Jii = np.zeros((s0,s1))

        # note: the observation operator H is common for all dims
        self.H = np.zeros((k, dim))


    def advance_model(self, Ed, Ew, r, dt, mQ = None):
        """
        This model captures the moisture dynamics at each grid point independently.

        Ed - drying equilibrium [grid shape]
        Ew - wetting equilibrium [grid shape]
        r - rain intensity for time unit [grid shape] [mm/h]
        dt - integration step [scalar] [s]
        mQ - the proess noise matrix [shape dim x dim, common across grid points]
        """
        Tk = self.Tk
        k = Tk.shape[0]                 # number of fuel classes
        m_ext = self.m_ext
        equi = self.equi
        rlag = self.rlag
        model_ids = self.model_ids
        m_i = self.m_i
        mn_i = self.mn_i
        J = self.J
        Jii = self.Jii
        P2 = self.P2

        # add assimilated equilibrium difference, which is shared across spatial locations
        EdA = Ed + m_ext[:,:,k]
        EwA = Ew + m_ext[:,:,k]

        dS = m_ext[:,:,k+1]

        # initialize equilibrium with current fuel moisture values
        # for various conditions, these values will be overwritten
        # if the are not overwritten, the model is in the dead zone (= no change to moisture)
        rlag[:,:] = 0.0


        # re-initialize the Jacobian (in case we must recompute it)
        J[:] = 0.0
        J[:,:,k,k] = 1.0
        J[:,:,k+1,k+1] = 1.0

        # where rainfall is above threshold (spatially different), apply
        # saturation model, equi and rlag are specific to fuel type and
        # location
        for i in range(k):

            # initialize equilibrium with current moisture
            m_i[:,:] = m_ext[:,:,i]
            equi[:,:] = m_ext[:,:,i]
            model_ids[:] = 4

            # on grid locations where there is rain, modify equilibrium
            has_rain = r > self.r0
            no_rain = np.logical_not(has_rain)

            equi[has_rain] = self.S + dS[has_rain]
            rlag[has_rain] = 1.0 / self.Trk * (1.0 - np.exp(- (r[r > self.r0] - self.r0) / self.rk))
            model_ids[has_rain] = 3

            # equilibrium is selected according to current moisture level
            is_drying = np.logical_and(no_rain, equi > EdA)
            equi[is_drying] = EdA[is_drying]
            model_ids[is_drying] = 1

            is_wetting = np.logical_and(no_rain, equi < EwA)
            equi[is_wetting] = EwA[is_wetting]
            model_ids[is_wetting] = 2

            rlag[no_rain] = 1.0 / Tk[i]

            # select appropriate integration method according to change for each fuel 
            # and location
            change = dt * rlag
            big_change = change > 0.01
            mn_i[big_change] = m_i[big_change] + (equi[big_change] - m_i[big_change]) * (1.0 - np.exp(-change[big_change]))
            small_change = np.logical_not(big_change)
            mn_i[small_change] = m_i[small_change] + (equi[small_change] - m_i[small_change]) * change[small_change] * (1.0 - 0.5 * change[small_change])

            # store in state matrix
            self.m_ext[:,:,i] = mn_i

            # update model state covariance if requested using the old state (the jacobian must be computed as well)
            if mQ is not None:

               # partial m_i/partial m_i
                Jii[big_change] = np.exp(-change[big_change])
                Jii[small_change] = 1.0 - change[small_change] * (1.0 - 0.5 * change[small_change])
                Jii[model_ids == 4] = 1.0
                J[:,:,i,i] = Jii

                # partial E/partial m_i
                Jii[:] = 0.0
                norain_big = np.logical_and(no_rain, big_change)
                norain_small = np.logical_and(no_rain, small_change)
                Jii[norain_big] = 1.0 - np.exp(-change[norain_big])
                Jii[norain_small] = change[norain_small] * (1.0 - 0.5 * change[norain_small])
                J[:,:,i,k] = Jii

                # partial S/partial m_i
                Jii[:] = 0.0
                rain_big = np.logical_and(has_rain, big_change)
                rain_small = np.logical_and(has_rain, small_change)
                Jii[rain_big] = 1.0 - np.exp(-change[rain_big])
                Jii[rain_small] = change[rain_small] * (1.0 - 0.5 * change[rain_small])
                J[:,:,i,k+1] = Jii


        # transformed to run in-place with one pre-allocated temporary
        for s in np.ndindex(self.P[:,:,0,0].shape):
            # P[s,:,:] = Js P[s,:,:] Js^T
            Ps = self.P[s[0],s[1],:,:]
            Js = J[s[0],s[1],:,:]
            np.dot(Js, Ps, P2)
            np.dot(P2, Js.T, Ps)

            # add process noise
            self.P[s[0],s[1],:,:] = Ps + mQ


    def get_state(self):
        """
        Return the current state for all grid points. READ-ONLY under normal circumstances.
        """
        return self.m_ext


    def get_state_covar(self):
        """
        Return the state covariance for all grid points. READ-ONLY under normal circumstances.
        """
        return self.P


    def kalman_update(self, O, V, fuel_types, Kg = None):
        """
        Updates the state using the observation at the grid point.

          O - the observations [shape grid_size x len(fuel_types)]
          V - the (diagonal) variance matrix of the measurements [shape grid_size x len(fuel_types) x len(fuel_types)]
          fuel_types - the fuel types for which the observations exist (used to construct observation vector)

        """
        P, H, P2 = self.P, self.H, self.P2
        H[:] = 0.0

        # construct an observation operator tailored to observed fuel types 
        for i in fuel_types:
            H[i,i] = 1.0
        Ho = H[fuel_types,:]

        for s in np.ndindex(P[:,:,0,0].shape):
          # re-use P2 to store forecast covariance for position s
          P2[:,:] = P[s[0],s[1],:,:]

          # compute Kalman gain
          I = np.dot(np.dot(Ho, P2), Ho.T) + V[s[0],s[1],:,:]
          K = np.dot(np.dot(P2, Ho.T), np.linalg.inv(I))

          # update state and state covariance
          self.m_ext[s[0],s[1],:] += np.dot(K, O[s[0],s[1],:] - self.m_ext[s[0],s[1],fuel_types])
          P[s[0],s[1],:,:] -= np.dot(np.dot(K, Ho), P2)

          if Kg is not None:
              Kg[s[0],s[1],:] = K[:,0]

