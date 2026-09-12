# Copyright 2023 The Ravens Authors. Licensed under the Apache License, Version 2.0.
"""Pick-and-place motion primitive (ravens PickPlace) with a record of what happened."""
import numpy as np

from see2act.sim import utils


class PickPlace:
    """Move above the pick pose, descend until contact, suction, lift, transfer, descend until contact, release.

    After each call, `self.last` holds a dict describing the execution (contacts, grasp success, timeouts);
    the evaluation uses it to categorize failures.
    """

    def __init__(self, height=0.32, speed=0.01, max_descent=10000):
        self.height, self.speed, self.max_descent = height, speed, max_descent
        self.last = None

    def __call__(self, movej, movep, ee, pose0, pose1):
        rec = dict(pick_contact=False, pick_attempts=0, pick_success=False, place_contact=False,
                   place_attempts=0, timeout=False, released=False)
        self.last = rec
        pick_pose, place_pose = pose0, pose1
        prepick_to_pick = ((0, 0, 0.32), (0, 0, 0, 1))
        postpick_to_pick = ((0, 0, self.height), (0, 0, 0, 1))
        prepick_pose = utils.multiply(pick_pose, prepick_to_pick)
        postpick_pose = utils.multiply(pick_pose, postpick_to_pick)
        timeout = movep(prepick_pose)

        delta = (np.float32([0, 0, -0.001]), utils.eulerXYZ_to_quatXYZW((0, 0, 0)))
        targ_pose = prepick_pose
        while not ee.detect_contact():
            targ_pose = utils.multiply(targ_pose, delta)
            timeout |= movep(targ_pose)
            rec["pick_attempts"] += 1
            if timeout or rec["pick_attempts"] >= self.max_descent:
                rec["timeout"] = True
                return True
        rec["pick_contact"] = True

        ee.activate()
        timeout |= movep(postpick_pose, self.speed)
        pick_success = ee.check_grasp()
        rec["pick_success"] = bool(pick_success)

        if pick_success:
            preplace_to_place = ((0, 0, self.height), (0, 0, 0, 1))
            postplace_to_place = ((0, 0, 0.32), (0, 0, 0, 1))
            preplace_pose = utils.multiply(place_pose, preplace_to_place)
            postplace_pose = utils.multiply(place_pose, postplace_to_place)
            targ_pose = preplace_pose
            while not ee.detect_contact():
                targ_pose = utils.multiply(targ_pose, delta)
                timeout |= movep(targ_pose, self.speed)
                rec["place_attempts"] += 1
                if timeout:
                    rec["timeout"] = True
                    return True
            rec["place_contact"] = True
            ee.release()
            rec["released"] = True
            timeout |= movep(postplace_pose)
        else:
            ee.release()
            timeout |= movep(prepick_pose)
        rec["timeout"] = bool(timeout)
        return timeout
