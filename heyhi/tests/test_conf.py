"""Tests for HeyHi configuration.

Run with nosetests.
"""
import pathlib
import unittest

import heyhi.conf


class TestRootConf(unittest.TestCase):
    def _load(self, overrides):
        data_dir = pathlib.Path(__file__).parent / "data"
        root_cfg = data_dir / "root.prototxt"

        task, meta_cfg = heyhi.conf.load_cfg(root_cfg, overrides)
        self.assertEqual(task, "test")
        return getattr(meta_cfg, task)

    def testLoadSimple(self):
        cfg = self._load(overrides=[])
        self.assertEqual(cfg.scalar, -1)
        self.assertEqual(cfg.sub.subscalar, -1)

    def testScalarOverride(self):
        cfg = self._load(overrides=["scalar=20"])
        self.assertEqual(cfg.scalar, 20)
        self.assertEqual(cfg.sub.subscalar, -1)

    def testScalarOverrideEnum(self):
        cfg = self._load(overrides=["enum_value=ONE"])
        self.assertEqual(cfg.enum_value, 1)

    def testScalarOverrideFloat(self):
        cfg = self._load(overrides=["scalar=20.5"])
        self.assertEqual(cfg.scalar, 20.5)
        self.assertEqual(cfg.sub.subscalar, -1)

    def testScalarOverrideSub(self):
        cfg = self._load(overrides=["sub.subscalar=20"])
        self.assertEqual(cfg.scalar, -1)
        self.assertEqual(cfg.sub.subscalar, 20)

    def testScalarOverrideTypeCast(self):
        self.assertRaises(ValueError, lambda: self._load(overrides=["sub.subscalar=20.5"]))

    def testIncludeScalar(self):
        cfg = self._load(overrides=["I=redefine_scalar"])
        self.assertEqual(cfg.scalar, 1.0)
        self.assertEqual(cfg.sub.subscalar, -1)

    def testIncludeMessage(self):
        cfg = self._load(overrides=["I=redefine_message"])
        self.assertEqual(cfg.scalar, -1)
        self.assertEqual(cfg.sub.subscalar, 22)
        self.assertEqual(cfg.sub2.subscalar, -1)

    def testIncludeMessageInside(self):
        cfg = self._load(overrides=["I.sub=redefine_subscalar_22"])
        self.assertEqual(cfg.scalar, -1)
        self.assertEqual(cfg.sub.subscalar, 22)
        self.assertEqual(cfg.sub2.subscalar, -1)

    def testIncludeMessageInsideTwice(self):
        cfg = self._load(overrides=["I.sub=redefine_subscalar_22", "I.sub2=redefine_subscalar_22"])
        self.assertEqual(cfg.scalar, -1)
        self.assertEqual(cfg.sub.subscalar, 22)
        self.assertEqual(cfg.sub2.subscalar, 22)


class TestRootWithIncludesConf(unittest.TestCase):
    def _load(self, overrides):
        data_dir = pathlib.Path(__file__).parent / "data"
        root_cfg = data_dir / "root_with_includes.prototxt"
        task, meta_cfg = heyhi.conf.load_cfg(root_cfg, overrides)
        self.assertEqual(task, "test")
        return getattr(meta_cfg, task)

    def testLoadSimple(self):
        cfg = self._load(overrides=[])
        self.assertEqual(cfg.scalar, -1)
        self.assertEqual(cfg.sub.subscalar, -1)
        self.assertEqual(cfg.sub2.subscalar, 22)

    def testLoadOverrideInclude(self):
        cfg = self._load(overrides=["I.sub2=redefine_subscalar_37"])
        self.assertEqual(cfg.scalar, -1)
        self.assertEqual(cfg.sub.subscalar, -1)
        self.assertEqual(cfg.sub2.subscalar, 37)


class TestRootWithIncludesAndRedefinesConf(unittest.TestCase):
    def _load(self, overrides):
        data_dir = pathlib.Path(__file__).parent / "data"
        root_cfg = data_dir / "root_with_includes_redefined.prototxt"
        task, meta_cfg = heyhi.conf.load_cfg(root_cfg, overrides)
        self.assertEqual(task, "test")
        return getattr(meta_cfg, task)

    def testLoadSimple(self):
        cfg = self._load(overrides=[])
        self.assertEqual(cfg.scalar, -1)
        self.assertEqual(cfg.sub.subscalar, 22)
        self.assertEqual(cfg.sub2.subscalar, 99)
