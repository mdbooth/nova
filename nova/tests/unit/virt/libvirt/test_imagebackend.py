# Copyright 2012 Grid Dynamics
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import base64
import contextlib
import inspect
import os
import shutil
import tempfile

from castellan import key_manager
import fixtures
import mock
from oslo_utils import units
from oslo_utils import uuidutils

import nova.conf
from nova import context
from nova import exception
from nova import objects
from nova import test
from nova.tests.unit import fake_processutils
from nova.tests.unit.virt.libvirt import fake_libvirt_utils
from nova.virt.image import model as imgmodel
from nova.virt import images
from nova.virt.libvirt import config as vconfig
from nova.virt.libvirt import imagebackend
from nova.virt.libvirt import imagecache
from nova.virt.libvirt.storage import rbd_utils

CONF = nova.conf.CONF


class FakeSecret(object):

    def value(self):
        return base64.b64decode("MTIzNDU2Cg==")


class FakeConn(object):

    def secretLookupByUUIDString(self, uuid):
        return FakeSecret()


class _ImageTestCase(object):

    def setUp(self):
        super(_ImageTestCase, self).setUp()
        self.INSTANCES_PATH = tempfile.mkdtemp(suffix='instances')
        self.flags(instances_path=self.INSTANCES_PATH)
        self.INSTANCE = objects.Instance(id=1, uuid=uuidutils.generate_uuid())
        self.DISK_INFO_PATH = os.path.join(self.INSTANCES_PATH,
                                           self.INSTANCE['uuid'], 'disk.info')
        self.NAME = 'fake.vm'
        self.CONTEXT = context.get_admin_context()
        self.PATH = os.path.join(
            fake_libvirt_utils.get_instance_path(self.INSTANCE), self.NAME)
        self.useFixture(fixtures.MonkeyPatch(
            'nova.virt.libvirt.imagebackend.libvirt_utils',
            fake_libvirt_utils))

    def tearDown(self):
        super(_ImageTestCase, self).tearDown()
        shutil.rmtree(self.INSTANCES_PATH)

    def test_libvirt_fs_info(self):
        image = self.image_class(self.INSTANCE, self.NAME)
        fs = image.libvirt_fs_info("/mnt")
        # check that exception hasn't been raised and the method
        # returned correct object
        self.assertIsInstance(fs, vconfig.LibvirtConfigGuestFilesys)
        self.assertEqual(fs.target_dir, "/mnt")
        if image.is_block_dev:
            self.assertEqual(fs.source_type, "block")
            self.assertEqual(fs.source_dev, image.path)
        else:
            self.assertEqual(fs.source_type, "file")
            self.assertEqual(fs.source_file, image.path)

    def test_libvirt_info(self):
        image = self.image_class(self.INSTANCE, self.NAME)
        extra_specs = {
            'quota:disk_read_bytes_sec': 10 * units.Mi,
            'quota:disk_read_iops_sec': 1 * units.Ki,
            'quota:disk_write_bytes_sec': 20 * units.Mi,
            'quota:disk_write_iops_sec': 2 * units.Ki,
            'quota:disk_total_bytes_sec': 30 * units.Mi,
            'quota:disk_total_iops_sec': 3 * units.Ki,
        }

        disk = image.libvirt_info(disk_bus="virtio",
                                  disk_dev="/dev/vda",
                                  device_type="cdrom",
                                  cache_mode="none",
                                  extra_specs=extra_specs,
                                  hypervisor_version=4004001)

        self.assertIsInstance(disk, vconfig.LibvirtConfigGuestDisk)
        self.assertEqual("/dev/vda", disk.target_dev)
        self.assertEqual("virtio", disk.target_bus)
        self.assertEqual("none", disk.driver_cache)
        self.assertEqual("cdrom", disk.source_device)

        self.assertEqual(10 * units.Mi, disk.disk_read_bytes_sec)
        self.assertEqual(1 * units.Ki, disk.disk_read_iops_sec)
        self.assertEqual(20 * units.Mi, disk.disk_write_bytes_sec)
        self.assertEqual(2 * units.Ki, disk.disk_write_iops_sec)
        self.assertEqual(30 * units.Mi, disk.disk_total_bytes_sec)
        self.assertEqual(3 * units.Ki, disk.disk_total_iops_sec)

    @mock.patch('nova.virt.disk.api.get_disk_size')
    def test_get_disk_size(self, get_disk_size):
        get_disk_size.return_value = 2361393152

        image = self.image_class(self.INSTANCE, self.NAME)
        self.assertEqual(2361393152, image.get_disk_size(image.path))
        get_disk_size.assert_called_once_with(image.path)

    def _flavor_disk_larger_than_image(self, mock_cache, path,
                                       file_format=None, disk_size=None):
        # flavor root disk size (100) > virtual disk size (99)
        size, virtual_size = 100, 99
        image = self.image_class(self.INSTANCE, self.NAME)
        mock_cache.return_value = imagecache.CachedImageInfo(
            path, file_format, virtual_size, disk_size)
        return image, size

    def _flavor_disk_smaller_than_image(self, mock_cache, path,
                                        file_format=None, disk_size=None):
        # flavor root disk size (99) < virtual disk size (100)
        size, virtual_size = 99, 100
        image = self.image_class(self.INSTANCE, self.NAME)
        mock_cache.return_value = imagecache.CachedImageInfo(
            path, file_format, virtual_size, disk_size)
        return image, size


class FlatTestCase(_ImageTestCase, test.NoDBTestCase):

    def setUp(self):
        self.image_class = imagebackend.Flat
        super(FlatTestCase, self).setUp()

    @mock.patch.object(fake_libvirt_utils, 'copy_image')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_func_output_path')
    def test_create_from_func(self, mock_cache, mock_copy):
        CONF.set_override('preallocate_images', 'space')
        image = self.image_class(self.INSTANCE, self.NAME)
        mock_cache.return_value = mock.sentinel.path
        size = mock.sentinel.size
        with self._create_mocks(image, size, should_resize=False):
            image.create_from_func(
                self.CONTEXT, mock.sentinel.func, mock.sentinel.cache_name,
                size, mock.sentinel.fallback)
            mock_copy.assert_called_once_with(mock.sentinel.path, self.PATH)

    @mock.patch.object(fake_libvirt_utils, 'copy_image')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_image_info')
    def test_create_from_image_success(self, mock_cache, mock_copy):
        CONF.set_override('preallocate_images', 'space')
        image, size = self._flavor_disk_larger_than_image(
            mock_cache, mock.sentinel.path)
        with self._create_mocks(image, size, should_resize=True):
            image.create_from_image(self.CONTEXT, mock.sentinel.image_id, size)
            mock_copy.assert_called_once_with(mock.sentinel.path, self.PATH)

    @mock.patch.object(fake_libvirt_utils, 'copy_image')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_image_info')
    def test_create_from_image_error(self, mock_cache, mock_copy):
        CONF.set_override('preallocate_images', 'space')
        image, size = self._flavor_disk_smaller_than_image(
            mock_cache, mock.sentinel.path)
        self.assertRaises(
            exception.FlavorDiskSmallerThanImage, image.create_from_image,
            self.CONTEXT, mock.sentinel.image_id, size)
        self.assertEqual(0, mock_copy.call_count)

    @contextlib.contextmanager
    def _create_mocks(self, image, size, should_resize):
        self.image_class.can_fallocate = None
        fallocate_calls = [
            'fallocate -l 1 %s.fallocate_test' % self.PATH,
            'fallocate -n -l %s %s' % (size, self.PATH),
        ]
        with test.nested(
            mock.patch('nova.virt.disk.api.extend'),
            mock.patch.object(image, 'correct_format'),
        ) as (mock_resize, mock_correct):
            fake_processutils.fake_execute_clear_log()
            fake_processutils.stub_out_processutils_execute(self.stubs)
            yield
            if should_resize:
                self.assertEqual(1, mock_resize.call_count)
            else:
                self.assertEqual(0, mock_resize.call_count)
            self.assertEqual(
                fallocate_calls, fake_processutils.fake_execute_get_log())
            self.assertEqual(1, mock_correct.call_count)

    def test_correct_format(self):
        self.stubs.UnsetAll()

        self.mox.StubOutWithMock(os.path, 'exists')
        self.mox.StubOutWithMock(imagebackend.images, 'qemu_img_info')

        os.path.exists(self.PATH).AndReturn(True)
        os.path.exists(self.DISK_INFO_PATH).AndReturn(False)
        info = self.mox.CreateMockAnything()
        info.file_format = 'foo'
        imagebackend.images.qemu_img_info(self.PATH).AndReturn(info)
        os.path.exists(CONF.instances_path).AndReturn(True)
        self.mox.ReplayAll()

        image = self.image_class(self.INSTANCE, self.NAME, path=self.PATH)
        self.assertEqual(image.driver_format, 'foo')

        self.mox.VerifyAll()

    @mock.patch.object(images, 'qemu_img_info',
                       side_effect=exception.InvalidDiskInfo(
                           reason='invalid path'))
    def test_resolve_driver_format(self, fake_qemu_img_info):
        image = self.image_class(self.INSTANCE, self.NAME)
        driver_format = image.resolve_driver_format()
        self.assertEqual(driver_format, 'raw')

    def test_get_model(self):
        image = self.image_class(self.INSTANCE, self.NAME)
        model = image.get_model(FakeConn())
        self.assertEqual(imgmodel.LocalFileImage(self.PATH,
                                                 imgmodel.FORMAT_RAW),
                         model)


class Qcow2TestCase(_ImageTestCase, test.NoDBTestCase):

    def setUp(self):
        self.image_class = imagebackend.Qcow2
        super(Qcow2TestCase, self).setUp()

    @mock.patch.object(fake_libvirt_utils, 'get_disk_backing_file')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_func_output_path')
    def test_check_backing_from_func_backless(self, mock_cache, mock_backing):
        mock_backing.return_value = None
        image = self.image_class(self.INSTANCE, self.NAME)
        image.check_backing_from_func(
            mock.sentinel.func, mock.sentinel.cache_name,
            mock.sentinel.fallback)
        self.assertEqual(0, mock_cache.call_count)

    @mock.patch.object(fake_libvirt_utils, 'get_disk_backing_file')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_func_output_path')
    def test_check_backing_from_func_exists(self, mock_cache, mock_backing):
        mock_backing.return_value = mock.sentinel.backing_file_path
        image = self.image_class(self.INSTANCE, self.NAME)
        with mock.patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            image.check_backing_from_func(
                mock.sentinel.func, mock.sentinel.cache_name,
                mock.sentinel.fallback)
        self.assertEqual(0, mock_cache.call_count)

    @mock.patch.object(fake_libvirt_utils, 'get_disk_backing_file')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_func_output_path')
    def test_check_backing_from_func_missing(self, mock_cache, mock_backing):
        mock_backing.return_value = mock.sentinel.backing_file_path
        image = self.image_class(self.INSTANCE, self.NAME)
        with mock.patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            image.check_backing_from_func(
                mock.sentinel.func, mock.sentinel.cache_name,
                mock.sentinel.fallback)
        mock_cache.assert_called_once_with(
            mock.sentinel.func, mock.sentinel.cache_name,
            mock.sentinel.fallback)

    @mock.patch.object(fake_libvirt_utils, 'get_disk_backing_file')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_image_info')
    def test_check_backing_from_image_backless(self, mock_cache, mock_backing):
        mock_backing.return_value = None
        image = self.image_class(self.INSTANCE, self.NAME)
        image.check_backing_from_image(
            self.CONTEXT, mock.sentinel.image_id, mock.sentinel.fallback)
        self.assertEqual(0, mock_cache.call_count)

    @mock.patch.object(fake_libvirt_utils, 'get_disk_backing_file')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_image_info')
    def test_check_backing_from_image_exists(self, mock_cache, mock_backing):
        mock_backing.return_value = mock.sentinel.backing_file_path
        image = self.image_class(self.INSTANCE, self.NAME)
        with mock.patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            image.check_backing_from_image(
                self.CONTEXT, mock.sentinel.image_id, mock.sentinel.fallback)
        self.assertEqual(0, mock_cache.call_count)

    @mock.patch.object(fake_libvirt_utils, 'get_disk_backing_file')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_image_info')
    def test_check_backing_from_image_missing(self, mock_cache, mock_backing):
        mock_backing.return_value = mock.sentinel.backing_file_path
        image = self.image_class(self.INSTANCE, self.NAME)
        with mock.patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            image.check_backing_from_image(
                self.CONTEXT, mock.sentinel.image_id, mock.sentinel.fallback)
        mock_cache.assert_called_once_with(
            self.CONTEXT, mock.sentinel.image_id, mock.sentinel.fallback)

    @mock.patch.object(fake_libvirt_utils, 'create_cow_image')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_func_output_path')
    def test_create_from_func(self, mock_cache, mock_create):
        CONF.set_override('preallocate_images', 'space')
        image = self.image_class(self.INSTANCE, self.NAME)
        mock_cache.return_value = mock.sentinel.path
        size = mock.sentinel.size
        with self._create_mocks(image, size, should_resize=False):
            image.create_from_func(
                self.CONTEXT, mock.sentinel.func, mock.sentinel.cache_name,
                size, mock.sentinel.fallback)
            mock_create.assert_called_once_with(mock.sentinel.path, self.PATH)

    @mock.patch.object(fake_libvirt_utils, 'create_cow_image')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_image_info')
    def test_create_from_image_success(self, mock_cache, mock_create):
        CONF.set_override('preallocate_images', 'space')
        image, size = self._flavor_disk_larger_than_image(
            mock_cache, mock.sentinel.path)
        with self._create_mocks(image, size, should_resize=True):
            image.create_from_image(self.CONTEXT, mock.sentinel.image_id, size)
            mock_create.assert_called_once_with(mock.sentinel.path, self.PATH)

    @mock.patch.object(fake_libvirt_utils, 'create_cow_image')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_image_info')
    def test_create_from_image_error(self, mock_cache, mock_create):
        CONF.set_override('preallocate_images', 'space')
        image, size = self._flavor_disk_smaller_than_image(
            mock_cache, mock.sentinel.path)
        self.assertRaises(
            exception.FlavorDiskSmallerThanImage, image.create_from_image,
            self.CONTEXT, mock.sentinel.image_id, size)
        self.assertEqual(0, mock_create.call_count)

    @contextlib.contextmanager
    def _create_mocks(self, image, size, should_resize):
        self.image_class.can_fallocate = None
        fallocate_calls = [
            'fallocate -l 1 %s.fallocate_test' % self.PATH,
            'fallocate -n -l %s %s' % (size, self.PATH),
        ]
        with mock.patch('nova.virt.disk.api.extend') as mock_resize:
            fake_processutils.fake_execute_clear_log()
            fake_processutils.stub_out_processutils_execute(self.stubs)
            yield
            if should_resize:
                self.assertEqual(1, mock_resize.call_count)
            else:
                self.assertEqual(0, mock_resize.call_count)
            self.assertEqual(
                fallocate_calls, fake_processutils.fake_execute_get_log())

    def test_resolve_driver_format(self):
        image = self.image_class(self.INSTANCE, self.NAME)
        driver_format = image.resolve_driver_format()
        self.assertEqual(driver_format, 'qcow2')

    def test_get_model(self):
        image = self.image_class(self.INSTANCE, self.NAME)
        model = image.get_model(FakeConn())
        self.assertEqual(imgmodel.LocalFileImage(self.PATH,
                                                 imgmodel.FORMAT_QCOW2),
                        model)


class LvmTestCase(_ImageTestCase, test.NoDBTestCase):
    VG = 'FakeVG'

    def setUp(self):
        self.image_class = imagebackend.Lvm
        super(LvmTestCase, self).setUp()
        self.flags(images_volume_group=self.VG, group='libvirt')
        self.flags(enabled=False, group='ephemeral_storage_encryption')
        self.INSTANCE['ephemeral_key_uuid'] = None
        self.LV = '%s_%s' % (self.INSTANCE['uuid'], self.NAME)
        self.PATH = os.path.join('/dev', self.VG, self.LV)

    @mock.patch.object(images, 'convert_image_unsafe')
    @mock.patch.object(imagebackend.lvm, 'create_volume')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_func_output_path')
    def test_create_from_func(self, mock_cache, mock_create, mock_convert):
        image = self.image_class(self.INSTANCE, self.NAME)
        mock_cache.return_value = mock.sentinel.path
        with self._create_mocks(should_resize=False):
            image.create_from_func(
                self.CONTEXT, mock.sentinel.func, mock.sentinel.cache_name,
                mock.sentinel.size, mock.sentinel.fallback)
            mock_create.assert_called_once_with(
                self.VG, self.LV, mock.sentinel.size, sparse=False)
            mock_convert.assert_called_once_with(
                mock.sentinel.path, self.PATH, 'raw', run_as_root=True)

    @mock.patch.object(images, 'convert_image')
    @mock.patch.object(imagebackend.lvm, 'create_volume')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_image_info')
    def test_create_from_image_success(self, mock_cache, mock_create,
                                       mock_convert):
        image, size = self._flavor_disk_larger_than_image(
            mock_cache, mock.sentinel.path, mock.sentinel.file_format)
        with self._create_mocks(should_resize=True):
            image.create_from_image(self.CONTEXT, mock.sentinel.image_id, size)
            mock_create.assert_called_once_with(
                self.VG, self.LV, size, sparse=False)
            mock_convert.assert_called_once_with(
                mock.sentinel.path, self.PATH, mock.sentinel.file_format,
                'raw', run_as_root=True)

    @mock.patch.object(imagebackend.lvm, 'create_volume')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_image_info')
    def test_create_from_image_error(self, mock_cache, mock_create):
        image, size = self._flavor_disk_smaller_than_image(
            mock_cache, mock.sentinel.path, mock.sentinel.file_format)
        self.assertRaises(
            exception.FlavorDiskSmallerThanImage, image.create_from_image,
            self.CONTEXT, mock.sentinel.image_id, size)
        self.assertEqual(0, mock_create.call_count)

    @contextlib.contextmanager
    def _create_mocks(self, should_resize):
        with test.nested(
            mock.patch('nova.virt.disk.api.resize2fs'),
            mock.patch.object(imagebackend.dmcrypt, 'create_volume')
        ) as (mock_resize, mock_dmcrypt_create):
            yield
            if should_resize:
                self.assertEqual(1, mock_resize.call_count)
            else:
                self.assertEqual(0, mock_resize.call_count)
            self.assertEqual(0, mock_dmcrypt_create.call_count)


class EncryptedLvmTestCase(_ImageTestCase, test.NoDBTestCase):
    VG = 'FakeVG'

    def setUp(self):
        super(EncryptedLvmTestCase, self).setUp()
        self.image_class = imagebackend.Lvm
        self.flags(enabled=True, group='ephemeral_storage_encryption')
        self.flags(cipher='aes-xts-plain64',
                   group='ephemeral_storage_encryption')
        self.flags(key_size=512, group='ephemeral_storage_encryption')
        self.flags(fixed_key='00000000000000000000000000000000'
                             '00000000000000000000000000000000',
                   group='key_manager')
        self.flags(images_volume_group=self.VG, group='libvirt')
        self.LV = '%s_%s' % (self.INSTANCE['uuid'], self.NAME)
        self.LV_PATH = os.path.join('/dev', self.VG, self.LV)
        self.PATH = os.path.join('/dev/mapper',
            imagebackend.dmcrypt.volume_name(self.LV))
        self.key_manager = key_manager.API()
        self.INSTANCE['ephemeral_key_uuid'] =\
            self.key_manager.create_key(self.CONTEXT, 'AES', 256)
        self.KEY = self.key_manager.get(self.CONTEXT,
            self.INSTANCE['ephemeral_key_uuid']).get_encoded()

    @mock.patch.object(imagebackend.lvm, 'remove_volumes')
    @mock.patch.object(imagebackend.lvm, 'create_volume')
    @mock.patch.object(imagebackend.dmcrypt, 'delete_volume')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_func_output_path')
    def test_get_encryption_key_error(self, mock_cache, mock_delete,
                                      mock_create, mock_remove):
        self.INSTANCE['ephemeral_key_uuid'] = 'bad key'
        image = self.image_class(self.INSTANCE, self.NAME)
        mock_cache.return_value = mock.sentinel.path
        self.assertRaises(
            KeyError, image.create_from_func, self.CONTEXT,
            mock.sentinel.func, mock.sentinel.cache_name, mock.sentinel.size)
        mock_remove.assert_called_once_with([self.LV_PATH])
        mock_delete.assert_called_once_with(self.PATH.rpartition('/')[2])

    @mock.patch.object(images, 'convert_image_unsafe')
    @mock.patch.object(imagebackend.lvm, 'create_volume')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_func_output_path')
    def test_create_from_func(self, mock_cache, mock_create, mock_convert):
        image = self.image_class(self.INSTANCE, self.NAME)
        mock_cache.return_value = mock.sentinel.path
        with self._create_mocks(should_resize=False):
            image.create_from_func(
                self.CONTEXT, mock.sentinel.func, mock.sentinel.cache_name,
                mock.sentinel.size, mock.sentinel.fallback)
            mock_create.assert_called_once_with(
                self.VG, self.LV, mock.sentinel.size, sparse=False)
            mock_convert.assert_called_once_with(
                mock.sentinel.path, self.PATH, 'raw', run_as_root=True)

    @mock.patch.object(images, 'convert_image')
    @mock.patch.object(imagebackend.lvm, 'create_volume')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_image_info')
    def test_create_from_image_success(self, mock_cache, mock_create,
                                       mock_convert):
        image, size = self._flavor_disk_larger_than_image(
            mock_cache, mock.sentinel.path, mock.sentinel.file_format)
        with self._create_mocks(should_resize=True):
            image.create_from_image(self.CONTEXT, mock.sentinel.image_id, size)
            mock_create.assert_called_once_with(
                self.VG, self.LV, size, sparse=False)
            mock_convert.assert_called_once_with(
                mock.sentinel.path, self.PATH, mock.sentinel.file_format,
                'raw', run_as_root=True)

    @mock.patch.object(imagebackend.lvm, 'create_volume')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_image_info')
    def test_create_from_image_error(self, mock_cache, mock_create):
        image, size = self._flavor_disk_smaller_than_image(
            mock_cache, mock.sentinel.path, mock.sentinel.file_format)
        self.assertRaises(
            exception.FlavorDiskSmallerThanImage, image.create_from_image,
            self.CONTEXT, mock.sentinel.image_id, size)
        self.assertEqual(0, mock_create.call_count)

    @contextlib.contextmanager
    def _create_mocks(self, should_resize):
        with test.nested(
            mock.patch('nova.virt.disk.api.resize2fs'),
            mock.patch.object(imagebackend.dmcrypt, 'create_volume')
        ) as (mock_resize, mock_dmcrypt_create):
            yield
            if should_resize:
                self.assertEqual(1, mock_resize.call_count)
            else:
                self.assertEqual(0, mock_resize.call_count)
            mock_dmcrypt_create.assert_called_once_with(
                self.PATH.rpartition('/')[2], self.LV_PATH,
                CONF.ephemeral_storage_encryption.cipher,
                CONF.ephemeral_storage_encryption.key_size, self.KEY)

    def test_get_model(self):
        image = self.image_class(self.INSTANCE, self.NAME)
        model = image.get_model(FakeConn())
        self.assertEqual(imgmodel.LocalBlockImage(self.PATH), model)


class RbdTestCase(_ImageTestCase, test.NoDBTestCase):
    FSID = "FakeFsID"
    POOL = "FakePool"
    USER = "FakeUser"
    CONF = "FakeConf"

    def setUp(self):
        self.image_class = imagebackend.Rbd
        super(RbdTestCase, self).setUp()
        self.flags(images_rbd_pool=self.POOL,
                   rbd_user=self.USER,
                   images_rbd_ceph_conf=self.CONF,
                   group='libvirt')
        self.mox.StubOutWithMock(rbd_utils, 'rbd')
        self.mox.StubOutWithMock(rbd_utils, 'rados')

    def test_remove_volume_on_error_exists_true(self):
        image = self.image_class(self.INSTANCE, self.NAME)
        name = '%s_%s' % (self.INSTANCE.uuid, self.NAME)

        @mock.patch.object(image, 'exists')
        @mock.patch.object(image.driver, 'remove_image')
        def _test(mock_remove, mock_exists):
            mock_exists.return_value = True
            try:
                with image.remove_volume_on_error():
                    raise test.TestingException('should re-raise')
            except test.TestingException:
                mock_remove.assert_called_once_with(name)
            else:
                raise Exception('Expected TestingException')
        _test()

    def test_remove_volume_on_error_exists_false(self):
        image = self.image_class(self.INSTANCE, self.NAME)

        @mock.patch.object(image, 'exists')
        @mock.patch.object(image.driver, 'remove_image')
        def _test(mock_remove, mock_exists):
            mock_exists.return_value = False
            try:
                with image.remove_volume_on_error():
                    raise test.TestingException('should re-raise')
            except test.TestingException:
                self.assertEqual(0, mock_remove.call_count)
            else:
                raise Exception('Expected TestingException')
        _test()

    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_func_output_path')
    @mock.patch.object(rbd_utils.RBDDriver, 'import_image')
    def test_create_from_func(self, mock_import, mock_cache):
        image = self.image_class(self.INSTANCE, self.NAME)
        name = '%s_%s' % (self.INSTANCE.uuid, self.NAME)
        mock_cache.return_value = mock.sentinel.path
        with self._resize_mock(image, should_resize=False):
            image.create_from_func(
                self.CONTEXT, mock.sentinel.func, mock.sentinel.cache_name,
                mock.sentinel.size, mock.sentinel.fallback)
            mock_import.assert_called_once_with(mock.sentinel.path, name)

    @mock.patch.object(rbd_utils.RBDDriver, 'import_image')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_image_info')
    def test_create_from_image_success(self, mock_cache, mock_import):
        name = '%s_%s' % (self.INSTANCE.uuid, self.NAME)
        image, size = self._flavor_disk_larger_than_image(
            mock_cache, mock.sentinel.path)
        with self._create_mocks(image):
            with self._resize_mock(image, should_resize=True):
                image.create_from_image(
                    self.CONTEXT, mock.sentinel.image_id, size)
                mock_import.assert_called_once_with(mock.sentinel.path, name)

    @mock.patch.object(rbd_utils.RBDDriver, 'import_image')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_image_info')
    def test_create_from_image_error(self, mock_cache, mock_import):
        image, size = self._flavor_disk_smaller_than_image(
            mock_cache, mock.sentinel.path)
        with self._create_mocks(image):
            self.assertRaises(
                exception.FlavorDiskSmallerThanImage, image.create_from_image,
                self.CONTEXT, mock.sentinel.image_id, size)
            self.assertEqual(0, mock_import.call_count)

    @mock.patch.object(rbd_utils.RBDDriver, 'clone')
    def test_create_from_image_clone_success(self, mock_clone):
        size = 100  # flavor root disk size (100) > virtual disk size (99)
        with self._clone_mocks(99) as [image, name, url]:
            image.create_from_image(self.CONTEXT, mock.sentinel.image_id, size)
            mock_clone.assert_called_once_with(url, name)

    @mock.patch.object(rbd_utils.RBDDriver, 'clone')
    @mock.patch.object(rbd_utils.RBDDriver, 'remove_image')
    def test_create_from_image_clone_error(self, mock_remove, mock_clone):
        size = 99  # flavor root disk size (99) < virtual disk size (100)
        with self._clone_mocks(100) as [image, name, url]:
            self.assertRaises(
                exception.FlavorDiskSmallerThanImage, image.create_from_image,
                self.CONTEXT, mock.sentinel.image_id, size)
            mock_clone.assert_called_once_with(url, name)
            mock_remove.assert_called_once_with(name)  # rollback on error

    @contextlib.contextmanager
    def _resize_mock(self, image, should_resize):
        with mock.patch.object(image.driver, 'resize') as mock_resize:
            yield
            if should_resize:
                self.assertEqual(1, mock_resize.call_count)
            else:
                self.assertEqual(0, mock_resize.call_count)

    @contextlib.contextmanager
    def _create_mocks(self, image):
        url1 = {'url': mock.sentinel.url1}
        url2 = {'url': mock.sentinel.url2}
        with test.nested(
            mock.patch.object(image.driver, 'is_cloneable'),
            mock.patch.object(imagebackend.IMAGE_API, 'get'),
        ) as (mock_cloneable, mock_get):
            mock_get.return_value = {'locations': [url1, url2]}
            mock_cloneable.side_effect = [False, False]  # none cloneable
            yield

    @contextlib.contextmanager
    def _clone_mocks(self, size):
        name = '%s_%s' % (self.INSTANCE.uuid, self.NAME)
        image = self.image_class(self.INSTANCE, self.NAME)
        url1 = {'url': mock.sentinel.url1}
        url2 = {'url': mock.sentinel.url2}
        with test.nested(
            mock.patch.object(image, 'exists'),
            mock.patch.object(image, 'get_disk_size'),
            mock.patch.object(image.driver, 'resize'),
            mock.patch.object(image.driver, 'is_cloneable'),
            mock.patch.object(imagebackend.IMAGE_API, 'get')
        ) as (mock_exists, mock_size, mock_resize, mock_cloneable, mock_get):
            mock_get.return_value = {'locations': [url1, url2]}
            mock_cloneable.side_effect = [False, True]  # 2nd is cloneable
            mock_size.return_value = size
            mock_exists.return_value = True
            yield image, name, url2
            self.assertEqual(0, mock_resize.call_count)

    def test_parent_compatible(self):
        self.assertEqual(inspect.getargspec(imagebackend.Image.libvirt_info),
                         inspect.getargspec(self.image_class.libvirt_info))

    def test_image_path(self):
        conf = "FakeConf"
        pool = "FakePool"
        user = "FakeUser"

        self.flags(images_rbd_pool=pool, group='libvirt')
        self.flags(images_rbd_ceph_conf=conf, group='libvirt')
        self.flags(rbd_user=user, group='libvirt')
        image = self.image_class(self.INSTANCE, self.NAME)
        rbd_path = "rbd:%s/%s:id=%s:conf=%s" % (pool, image.rbd_name,
                                                user, conf)

        self.assertEqual(image.path, rbd_path)

    def test_get_disk_size(self):
        image = self.image_class(self.INSTANCE, self.NAME)
        with mock.patch.object(image.driver, 'size') as size_mock:
            size_mock.return_value = 2361393152

            self.assertEqual(2361393152, image.get_disk_size(image.path))
            size_mock.assert_called_once_with(image.rbd_name)

    @mock.patch.object(rbd_utils.RBDDriver, "get_mon_addrs")
    def test_libvirt_info(self, mock_mon_addrs):
        def get_mon_addrs():
            hosts = ["server1", "server2"]
            ports = ["1899", "1920"]
            return hosts, ports
        mock_mon_addrs.side_effect = get_mon_addrs

        super(RbdTestCase, self).test_libvirt_info()

    @mock.patch.object(rbd_utils.RBDDriver, "get_mon_addrs")
    def test_get_model(self, mock_mon_addrs):
        pool = "FakePool"
        user = "FakeUser"

        self.flags(images_rbd_pool=pool, group='libvirt')
        self.flags(rbd_user=user, group='libvirt')
        self.flags(rbd_secret_uuid="3306a5c4-8378-4b3c-aa1f-7b48d3a26172",
                   group='libvirt')

        def get_mon_addrs():
            hosts = ["server1", "server2"]
            ports = ["1899", "1920"]
            return hosts, ports
        mock_mon_addrs.side_effect = get_mon_addrs

        image = self.image_class(self.INSTANCE, self.NAME)
        model = image.get_model(FakeConn())
        self.assertEqual(imgmodel.RBDImage(
            self.INSTANCE["uuid"] + "_fake.vm",
            "FakePool",
            "FakeUser",
            "MTIzNDU2Cg==",
            ["server1:1899", "server2:1920"]),
                         model)

    def test_import_file(self):
        image = self.image_class(self.INSTANCE, self.NAME)

        @mock.patch.object(image, 'exists')
        @mock.patch.object(image.driver, 'remove_image')
        @mock.patch.object(image.driver, 'import_image')
        def _test(mock_import, mock_remove, mock_exists):
            mock_exists.return_value = True
            image.import_file(self.INSTANCE, mock.sentinel.file,
                              mock.sentinel.remote_name)
            name = '%s_%s' % (self.INSTANCE.uuid,
                              mock.sentinel.remote_name)
            mock_exists.assert_called_once_with()
            mock_remove.assert_called_once_with(name)
            mock_import.assert_called_once_with(mock.sentinel.file, name)
        _test()

    def test_import_file_not_found(self):
        image = self.image_class(self.INSTANCE, self.NAME)

        @mock.patch.object(image, 'exists')
        @mock.patch.object(image.driver, 'remove_image')
        @mock.patch.object(image.driver, 'import_image')
        def _test(mock_import, mock_remove, mock_exists):
            mock_exists.return_value = False
            image.import_file(self.INSTANCE, mock.sentinel.file,
                              mock.sentinel.remote_name)
            name = '%s_%s' % (self.INSTANCE.uuid,
                              mock.sentinel.remote_name)
            mock_exists.assert_called_once_with()
            self.assertFalse(mock_remove.called)
            mock_import.assert_called_once_with(mock.sentinel.file, name)
        _test()

    def test_get_parent_pool(self):
        image = self.image_class(self.INSTANCE, self.NAME)
        with mock.patch.object(rbd_utils.RBDDriver, 'parent_info') as mock_pi:
            mock_pi.return_value = [self.POOL, 'fake-image', 'fake-snap']
            parent_pool = image._get_parent_pool(self.CONTEXT, 'fake-image',
                                                 self.FSID)
            self.assertEqual(self.POOL, parent_pool)

    def test_get_parent_pool_no_parent_info(self):
        image = self.image_class(self.INSTANCE, self.NAME)
        rbd_uri = 'rbd://%s/%s/fake-image/fake-snap' % (self.FSID, self.POOL)
        with test.nested(mock.patch.object(rbd_utils.RBDDriver, 'parent_info'),
                         mock.patch.object(imagebackend.IMAGE_API, 'get'),
                         ) as (mock_pi, mock_get):
            mock_pi.side_effect = exception.ImageUnacceptable(image_id='test',
                                                              reason='test')
            mock_get.return_value = {'locations': [{'url': rbd_uri}]}
            parent_pool = image._get_parent_pool(self.CONTEXT, 'fake-image',
                                                 self.FSID)
            self.assertEqual(self.POOL, parent_pool)

    def test_get_parent_pool_non_local_image(self):
        image = self.image_class(self.INSTANCE, self.NAME)
        rbd_uri = 'rbd://remote-cluster/remote-pool/fake-image/fake-snap'
        with test.nested(
                mock.patch.object(rbd_utils.RBDDriver, 'parent_info'),
                mock.patch.object(imagebackend.IMAGE_API, 'get')
        ) as (mock_pi, mock_get):
            mock_pi.side_effect = exception.ImageUnacceptable(image_id='test',
                                                              reason='test')
            mock_get.return_value = {'locations': [{'url': rbd_uri}]}
            self.assertRaises(exception.ImageUnacceptable,
                              image._get_parent_pool, self.CONTEXT,
                              'fake-image', self.FSID)

    def test_direct_snapshot(self):
        image = self.image_class(self.INSTANCE, self.NAME)
        test_snap = 'rbd://%s/%s/fake-image-id/snap' % (self.FSID, self.POOL)
        with test.nested(
                mock.patch.object(rbd_utils.RBDDriver, 'get_fsid',
                                  return_value=self.FSID),
                mock.patch.object(image, '_get_parent_pool',
                                  return_value=self.POOL),
                mock.patch.object(rbd_utils.RBDDriver, 'create_snap'),
                mock.patch.object(rbd_utils.RBDDriver, 'clone'),
                mock.patch.object(rbd_utils.RBDDriver, 'flatten'),
                mock.patch.object(image, 'cleanup_direct_snapshot')
        ) as (mock_fsid, mock_parent, mock_create_snap, mock_clone,
              mock_flatten, mock_cleanup):
            location = image.direct_snapshot(self.CONTEXT, 'fake-snapshot',
                                             'fake-format', 'fake-image-id',
                                             'fake-base-image')
            mock_fsid.assert_called_once_with()
            mock_parent.assert_called_once_with(self.CONTEXT,
                                                'fake-base-image',
                                                self.FSID)
            mock_create_snap.assert_has_calls([mock.call(image.rbd_name,
                                                         'fake-snapshot',
                                                         protect=True),
                                               mock.call('fake-image-id',
                                                         'snap',
                                                         pool=self.POOL,
                                                         protect=True)])
            mock_clone.assert_called_once_with(mock.ANY, 'fake-image-id',
                                               dest_pool=self.POOL)
            mock_flatten.assert_called_once_with('fake-image-id',
                                                 pool=self.POOL)
            mock_cleanup.assert_called_once_with(mock.ANY)
            self.assertEqual(test_snap, location)

    def test_direct_snapshot_cleans_up_on_failures(self):
        image = self.image_class(self.INSTANCE, self.NAME)
        test_snap = 'rbd://%s/%s/%s/snap' % (self.FSID, image.pool,
                                             image.rbd_name)
        with test.nested(
                mock.patch.object(rbd_utils.RBDDriver, 'get_fsid',
                                  return_value=self.FSID),
                mock.patch.object(image, '_get_parent_pool',
                                  return_value=self.POOL),
                mock.patch.object(rbd_utils.RBDDriver, 'create_snap'),
                mock.patch.object(rbd_utils.RBDDriver, 'clone',
                                  side_effect=exception.Forbidden('testing')),
                mock.patch.object(rbd_utils.RBDDriver, 'flatten'),
                mock.patch.object(image, 'cleanup_direct_snapshot')) as (
                mock_fsid, mock_parent, mock_create_snap, mock_clone,
                mock_flatten, mock_cleanup):
            self.assertRaises(exception.Forbidden, image.direct_snapshot,
                              self.CONTEXT, 'snap', 'fake-format',
                              'fake-image-id', 'fake-base-image')
            mock_create_snap.assert_called_once_with(image.rbd_name, 'snap',
                                                     protect=True)
            self.assertFalse(mock_flatten.called)
            mock_cleanup.assert_called_once_with(dict(url=test_snap))

    def test_cleanup_direct_snapshot(self):
        image = self.image_class(self.INSTANCE, self.NAME)
        test_snap = 'rbd://%s/%s/%s/snap' % (self.FSID, image.pool,
                                             image.rbd_name)
        with test.nested(
                mock.patch.object(rbd_utils.RBDDriver, 'remove_snap'),
                mock.patch.object(rbd_utils.RBDDriver, 'destroy_volume')
        ) as (mock_rm, mock_destroy):
            # Ensure that the method does nothing when no location is provided
            image.cleanup_direct_snapshot(None)
            self.assertFalse(mock_rm.called)

            # Ensure that destroy_volume is not called
            image.cleanup_direct_snapshot(dict(url=test_snap))
            mock_rm.assert_called_once_with(image.rbd_name, 'snap', force=True,
                                            ignore_errors=False,
                                            pool=image.pool)
            self.assertFalse(mock_destroy.called)

    def test_cleanup_direct_snapshot_destroy_volume(self):
        image = self.image_class(self.INSTANCE, self.NAME)
        test_snap = 'rbd://%s/%s/%s/snap' % (self.FSID, image.pool,
                                             image.rbd_name)
        with test.nested(
                mock.patch.object(rbd_utils.RBDDriver, 'remove_snap'),
                mock.patch.object(rbd_utils.RBDDriver, 'destroy_volume')
        ) as (mock_rm, mock_destroy):
            # Ensure that destroy_volume is called
            image.cleanup_direct_snapshot(dict(url=test_snap),
                                          also_destroy_volume=True)
            mock_rm.assert_called_once_with(image.rbd_name, 'snap',
                                            force=True,
                                            ignore_errors=False,
                                            pool=image.pool)
            mock_destroy.assert_called_once_with(image.rbd_name,
                                                 pool=image.pool)


class PloopTestCase(_ImageTestCase, test.NoDBTestCase):

    def setUp(self):
        self.image_class = imagebackend.Ploop
        super(PloopTestCase, self).setUp()

    @mock.patch.object(fake_libvirt_utils, 'copy_image')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_func_output_path')
    def test_create_from_func(self, mock_cache, mock_copy):
        target = os.path.join(self.PATH, 'root.hds')
        image = self.image_class(self.INSTANCE, self.NAME)
        mock_cache.return_value = mock.sentinel.path
        with self._create_mocks(should_resize=False):
            image.create_from_func(
                self.CONTEXT, mock.sentinel.func, mock.sentinel.cache_name,
                mock.sentinel.size, mock.sentinel.fallback)
            mock_copy.assert_called_once_with(mock.sentinel.path, target)

    @mock.patch.object(fake_libvirt_utils, 'copy_image')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_image_info')
    def test_create_from_image_success(self, mock_cache, mock_copy):
        target = os.path.join(self.PATH, 'root.hds')
        image, size = self._flavor_disk_larger_than_image(
            mock_cache, mock.sentinel.path)
        with self._create_mocks(should_resize=True):
            image.create_from_image(self.CONTEXT, mock.sentinel.image_id, size)
            mock_copy.assert_called_once_with(mock.sentinel.path, target)

    @mock.patch.object(fake_libvirt_utils, 'copy_image')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_image_info')
    def test_create_from_image_error(self, mock_cache, mock_copy):
        image, size = self._flavor_disk_smaller_than_image(
            mock_cache, mock.sentinel.path)
        self.assertRaises(
            exception.FlavorDiskSmallerThanImage, image.create_from_image,
            self.CONTEXT, mock.sentinel.image_id, size)
        self.assertEqual(0, mock_copy.call_count)

    @mock.patch.object(imagebackend.IMAGE_API, 'get')
    def test_create_from_image_format_raw(self, mock_get):
        CONF.set_override('force_raw_images', False)
        mock_get.return_value = {'disk_format': 'raw'}
        self._test_create_from_image_format_success()

    @mock.patch.object(imagebackend.IMAGE_API, 'get')
    def test_create_from_image_format_ploop(self, mock_get):
        CONF.set_override('force_raw_images', False)
        mock_get.return_value = {'disk_format': 'ploop'}
        self._test_create_from_image_format_success()

    @mock.patch.object(fake_libvirt_utils, 'copy_image')
    @mock.patch.object(imagecache.ImageCacheLocalDir, 'get_image_info')
    def _test_create_from_image_format_success(self, mock_cache, mock_copy):
        target = os.path.join(self.PATH, 'root.hds')
        image, size = self._flavor_disk_larger_than_image(
            mock_cache, mock.sentinel.path)
        with self._create_mocks(should_resize=True):
            image.create_from_image(self.CONTEXT, mock.sentinel.image_id, size)
            mock_copy.assert_called_once_with(mock.sentinel.path, target)

    @mock.patch.object(imagebackend.IMAGE_API, 'get')
    def test_create_from_image_format_error(self, mock_get):
        CONF.set_override('force_raw_images', False)
        mock_get.return_value = {'disk_format': mock.sentinel.bad_format}
        image = self.image_class(self.INSTANCE, self.NAME)
        self.assertRaises(
            exception.ImageUnacceptable, image.create_from_image, self.CONTEXT,
            mock.sentinel.image_id, mock.sentinel.size)

    @contextlib.contextmanager
    def _create_mocks(self, should_resize):
        with test.nested(
            mock.patch.object(imagebackend.Ploop, 'resize_image'),
            mock.patch.object(imagebackend.Ploop, '_restore_descriptor')
        ) as (mock_resize, mock_restore):
            yield
            self.assertEqual(1, mock_restore.call_count)
            if should_resize:
                self.assertEqual(1, mock_resize.call_count)
            else:
                self.assertEqual(0, mock_resize.call_count)


class BackendTestCase(test.NoDBTestCase):
    INSTANCE = objects.Instance(id=1, uuid=uuidutils.generate_uuid())
    NAME = 'fake-name.suffix'

    def setUp(self):
        super(BackendTestCase, self).setUp()
        self.flags(enabled=False, group='ephemeral_storage_encryption')
        self.INSTANCE['ephemeral_key_uuid'] = None

    def get_image(self, use_cow, image_type):
        return imagebackend.Backend(use_cow).image(self.INSTANCE,
                                                   self.NAME,
                                                   image_type)

    def _test_image(self, image_type, image_not_cow, image_cow):
        image1 = self.get_image(False, image_type)
        image2 = self.get_image(True, image_type)

        def assertIsInstance(instance, class_object):
            failure = ('Expected %s,' +
                       ' but got %s.') % (class_object.__name__,
                                          instance.__class__.__name__)
            self.assertIsInstance(instance, class_object, msg=failure)

        assertIsInstance(image1, image_not_cow)
        assertIsInstance(image2, image_cow)

    def test_image_flat(self):
        self._test_image('raw', imagebackend.Flat, imagebackend.Flat)

    def test_image_flat_preallocate_images(self):
        flags = ('space', 'Space', 'SPACE')
        for f in flags:
            self.flags(preallocate_images=f)
            raw = imagebackend.Flat(self.INSTANCE, 'fake_disk', '/tmp/xyz')
            self.assertTrue(raw.preallocate)

    def test_image_flat_preallocate_images_bad_conf(self):
        self.flags(preallocate_images='space1')
        raw = imagebackend.Flat(self.INSTANCE, 'fake_disk', '/tmp/xyz')
        self.assertFalse(raw.preallocate)

    def test_image_flat_native_io(self):
        self.flags(preallocate_images="space")
        raw = imagebackend.Flat(self.INSTANCE, 'fake_disk', '/tmp/xyz')
        self.assertEqual(raw.driver_io, "native")

    def test_image_qcow2(self):
        self._test_image('qcow2', imagebackend.Qcow2, imagebackend.Qcow2)

    def test_image_qcow2_preallocate_images(self):
        flags = ('space', 'Space', 'SPACE')
        for f in flags:
            self.flags(preallocate_images=f)
            qcow = imagebackend.Qcow2(self.INSTANCE, 'fake_disk', '/tmp/xyz')
            self.assertTrue(qcow.preallocate)

    def test_image_qcow2_preallocate_images_bad_conf(self):
        self.flags(preallocate_images='space1')
        qcow = imagebackend.Qcow2(self.INSTANCE, 'fake_disk', '/tmp/xyz')
        self.assertFalse(qcow.preallocate)

    def test_image_qcow2_native_io(self):
        self.flags(preallocate_images="space")
        qcow = imagebackend.Qcow2(self.INSTANCE, 'fake_disk', '/tmp/xyz')
        self.assertEqual(qcow.driver_io, "native")

    def test_image_lvm_native_io(self):
        def _test_native_io(is_sparse, driver_io):
            self.flags(images_volume_group='FakeVG', group='libvirt')
            self.flags(sparse_logical_volumes=is_sparse, group='libvirt')
            lvm = imagebackend.Lvm(self.INSTANCE, 'fake_disk')
            self.assertEqual(lvm.driver_io, driver_io)
        _test_native_io(is_sparse=False, driver_io="native")
        _test_native_io(is_sparse=True, driver_io=None)

    def test_image_lvm(self):
        self.flags(images_volume_group='FakeVG', group='libvirt')
        self._test_image('lvm', imagebackend.Lvm, imagebackend.Lvm)

    def test_image_rbd(self):
        conf = "FakeConf"
        pool = "FakePool"
        self.flags(images_rbd_pool=pool, group='libvirt')
        self.flags(images_rbd_ceph_conf=conf, group='libvirt')
        self.mox.StubOutWithMock(rbd_utils, 'rbd')
        self.mox.StubOutWithMock(rbd_utils, 'rados')
        self._test_image('rbd', imagebackend.Rbd, imagebackend.Rbd)

    def test_image_default(self):
        self._test_image('default', imagebackend.Flat, imagebackend.Qcow2)
