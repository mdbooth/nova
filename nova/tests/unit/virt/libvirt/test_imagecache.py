# Copyright 2012 Michael Still and Canonical Inc
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


import contextlib
import os
import six
import time

import mock
from oslo_concurrency import lockutils
from oslo_concurrency import processutils
from oslo_log import formatters
from oslo_log import log as logging
from oslo_utils import fileutils
from oslo_utils import importutils
from oslo_utils import uuidutils
from six.moves import cStringIO

from nova import conductor
import nova.conf
from nova import context
from nova import exception
from nova import objects
from nova import test
from nova.tests.unit import fake_instance
from nova import utils
from nova.virt import images
from nova.virt.libvirt import imagecache
from nova.virt.libvirt import utils as libvirt_utils

CONF = nova.conf.CONF


@contextlib.contextmanager
def intercept_log_messages():
    try:
        mylog = logging.getLogger('nova')
        stream = cStringIO()
        handler = logging.logging.StreamHandler(stream)
        handler.setFormatter(formatters.ContextFormatter())
        mylog.logger.addHandler(handler)
        yield stream
    finally:
        mylog.logger.removeHandler(handler)


class GetCacheFnameTestCase(test.NoDBTestCase):
    def test_get_cache_fname(self):
        # Ensure a known input to this function produces a known output.

        # This test assures us that, used in the expected manner, the function
        # doesn't raise an exception in either python2 or python3. It also
        # serves as a canary to warn if any change in underlying libraries
        # would produce output incompatible with current usage.

        # Take a known image_id and the pre-calculated hexdigest of its sha1
        image_id = 'fd0cb2f1-8375-44c9-b1f4-3e1f4c4a8ef0'
        expected_cache_name = '0d5e6b61602d758984b3bf038267614d6016eb2a'

        cache_name = imagecache.get_cache_fname(image_id)
        self.assertEqual(expected_cache_name, cache_name)


class ImageCacheManagerTestCase(test.NoDBTestCase):

    def setUp(self):
        super(ImageCacheManagerTestCase, self).setUp()
        self.stock_instance_names = set(['instance-00000001',
                                         'instance-00000002',
                                         'instance-00000003',
                                         'banana-42-hamster'])

    @mock.patch.object(os.path, 'exists', return_value=True)
    @mock.patch.object(time, 'time', return_value=2000000)
    @mock.patch.object(os.path, 'getmtime', return_value=1000000)
    def test_get_age_of_file(self, mock_getmtime, mock_time, mock_exists):
        image_cache_manager = imagecache.ImageCacheManager()
        exists, age = image_cache_manager._get_age_of_file('/tmp')
        self.assertTrue(exists)
        self.assertEqual(1000000, age)

    @mock.patch.object(os.path, 'exists', return_value=False)
    def test_get_age_of_file_not_exists(self, mock_exists):
        image_cache_manager = imagecache.ImageCacheManager()
        exists, age = image_cache_manager._get_age_of_file('/tmp')
        self.assertFalse(exists)
        self.assertEqual(0, age)

    def test_list_base_images(self):
        listing = ['00000001',
                   'ephemeral_0_20_None',
                   '17d1b00b81642842e514494a78e804e9a511637c_5368709120.info',
                   '00000004',
                   'swap_1000']
        images = ['e97222e91fc4241f49a7f520d1dcf446751129b3_sm',
                  'e09c675c2d1cfac32dae3c2d83689c8c94bc693b_sm',
                  'e97222e91fc4241f49a7f520d1dcf446751129b3',
                  '17d1b00b81642842e514494a78e804e9a511637c',
                  '17d1b00b81642842e514494a78e804e9a511637c_5368709120',
                  '17d1b00b81642842e514494a78e804e9a511637c_10737418240']
        listing.extend(images)

        self.stub_out('os.listdir', lambda x: listing)
        self.stub_out('os.path.isfile', lambda x: True)

        base_dir = '/var/lib/nova/instances/_base'
        self.flags(instances_path='/var/lib/nova/instances')

        image_cache_manager = imagecache.ImageCacheManager()
        image_cache_manager._scan_base_images(base_dir)

        sanitized = []
        for ent in image_cache_manager.unexplained_images:
            sanitized.append(ent.replace(base_dir + '/', ''))

        self.assertEqual(sorted(sanitized), sorted(images))

        expected = os.path.join(base_dir,
                                'e97222e91fc4241f49a7f520d1dcf446751129b3')
        self.assertIn(expected, image_cache_manager.unexplained_images)

        expected = os.path.join(base_dir,
                                '17d1b00b81642842e514494a78e804e9a511637c_'
                                '10737418240')
        self.assertIn(expected, image_cache_manager.unexplained_images)

        unexpected = os.path.join(base_dir, '00000004')
        self.assertNotIn(unexpected, image_cache_manager.unexplained_images)

        for ent in image_cache_manager.unexplained_images:
            self.assertTrue(ent.startswith(base_dir))

        self.assertEqual(len(image_cache_manager.originals), 2)

        expected = os.path.join(base_dir,
                                '17d1b00b81642842e514494a78e804e9a511637c')
        self.assertIn(expected, image_cache_manager.originals)

        unexpected = os.path.join(base_dir,
                                  '17d1b00b81642842e514494a78e804e9a511637c_'
                                '10737418240')
        self.assertNotIn(unexpected, image_cache_manager.originals)

        self.assertEqual(1, len(image_cache_manager.back_swap_images))
        self.assertIn('swap_1000', image_cache_manager.back_swap_images)

    def test_list_backing_images_small(self):
        self.stub_out('os.listdir',
                       lambda x: ['_base', 'instance-00000001',
                                  'instance-00000002', 'instance-00000003'])
        self.stub_out('os.path.exists',
                       lambda x: x.find('instance-') != -1)
        self.stubs.Set(libvirt_utils, 'get_disk_backing_file',
                       lambda x: 'e97222e91fc4241f49a7f520d1dcf446751129b3_sm')

        found = os.path.join(CONF.instances_path,
                             CONF.image_cache_subdirectory_name,
                             'e97222e91fc4241f49a7f520d1dcf446751129b3_sm')

        image_cache_manager = imagecache.ImageCacheManager()
        image_cache_manager.unexplained_images = [found]
        image_cache_manager.instance_names = self.stock_instance_names

        inuse_images = image_cache_manager._list_backing_images()

        self.assertEqual(inuse_images, [found])
        self.assertEqual(len(image_cache_manager.unexplained_images), 0)

    def test_list_backing_images_resized(self):
        self.stub_out('os.listdir',
                       lambda x: ['_base', 'instance-00000001',
                                  'instance-00000002', 'instance-00000003'])
        self.stub_out('os.path.exists',
                       lambda x: x.find('instance-') != -1)
        self.stubs.Set(libvirt_utils, 'get_disk_backing_file',
                       lambda x: ('e97222e91fc4241f49a7f520d1dcf446751129b3_'
                                  '10737418240'))

        found = os.path.join(CONF.instances_path,
                             CONF.image_cache_subdirectory_name,
                             'e97222e91fc4241f49a7f520d1dcf446751129b3_'
                             '10737418240')

        image_cache_manager = imagecache.ImageCacheManager()
        image_cache_manager.unexplained_images = [found]
        image_cache_manager.instance_names = self.stock_instance_names

        inuse_images = image_cache_manager._list_backing_images()

        self.assertEqual(inuse_images, [found])
        self.assertEqual(len(image_cache_manager.unexplained_images), 0)

    def test_list_backing_images_instancename(self):
        self.stub_out('os.listdir',
                       lambda x: ['_base', 'banana-42-hamster'])
        self.stub_out('os.path.exists',
                       lambda x: x.find('banana-42-hamster') != -1)
        self.stubs.Set(libvirt_utils, 'get_disk_backing_file',
                       lambda x: 'e97222e91fc4241f49a7f520d1dcf446751129b3_sm')

        found = os.path.join(CONF.instances_path,
                             CONF.image_cache_subdirectory_name,
                             'e97222e91fc4241f49a7f520d1dcf446751129b3_sm')

        image_cache_manager = imagecache.ImageCacheManager()
        image_cache_manager.unexplained_images = [found]
        image_cache_manager.instance_names = self.stock_instance_names

        inuse_images = image_cache_manager._list_backing_images()

        self.assertEqual(inuse_images, [found])
        self.assertEqual(len(image_cache_manager.unexplained_images), 0)

    def test_list_backing_images_disk_notexist(self):
        self.stub_out('os.listdir',
                       lambda x: ['_base', 'banana-42-hamster'])
        self.stub_out('os.path.exists',
                       lambda x: x.find('banana-42-hamster') != -1)

        def fake_get_disk(disk_path):
            raise processutils.ProcessExecutionError()

        self.stubs.Set(libvirt_utils, 'get_disk_backing_file', fake_get_disk)

        image_cache_manager = imagecache.ImageCacheManager()
        image_cache_manager.unexplained_images = []
        image_cache_manager.instance_names = self.stock_instance_names

        self.assertRaises(processutils.ProcessExecutionError,
                          image_cache_manager._list_backing_images)

    def test_find_base_file_nothing(self):
        self.stub_out('os.path.exists', lambda x: False)

        base_dir = '/var/lib/nova/instances/_base'
        fingerprint = '549867354867'
        image_cache_manager = imagecache.ImageCacheManager()
        res = list(image_cache_manager._find_base_file(base_dir, fingerprint))

        self.assertEqual(0, len(res))

    def test_find_base_file_small(self):
        fingerprint = '968dd6cc49e01aaa044ed11c0cce733e0fa44a6a'
        self.stub_out('os.path.exists',
                       lambda x: x.endswith('%s_sm' % fingerprint))

        base_dir = '/var/lib/nova/instances/_base'
        image_cache_manager = imagecache.ImageCacheManager()
        res = list(image_cache_manager._find_base_file(base_dir, fingerprint))

        base_file = os.path.join(base_dir, fingerprint + '_sm')
        self.assertEqual(res, [(base_file, True, False)])

    def test_find_base_file_resized(self):
        fingerprint = '968dd6cc49e01aaa044ed11c0cce733e0fa44a6a'
        listing = ['00000001',
                   'ephemeral_0_20_None',
                   '968dd6cc49e01aaa044ed11c0cce733e0fa44a6a_10737418240',
                   '00000004']

        self.stub_out('os.listdir', lambda x: listing)
        self.stub_out('os.path.exists',
                       lambda x: x.endswith('%s_10737418240' % fingerprint))
        self.stub_out('os.path.isfile', lambda x: True)

        base_dir = '/var/lib/nova/instances/_base'
        image_cache_manager = imagecache.ImageCacheManager()
        image_cache_manager._scan_base_images(base_dir)
        res = list(image_cache_manager._find_base_file(base_dir, fingerprint))

        base_file = os.path.join(base_dir, fingerprint + '_10737418240')
        self.assertEqual(res, [(base_file, False, True)])

    def test_find_base_file_all(self):
        fingerprint = '968dd6cc49e01aaa044ed11c0cce733e0fa44a6a'
        listing = ['00000001',
                   'ephemeral_0_20_None',
                   '968dd6cc49e01aaa044ed11c0cce733e0fa44a6a_sm',
                   '968dd6cc49e01aaa044ed11c0cce733e0fa44a6a_10737418240',
                   '00000004']

        self.stub_out('os.listdir', lambda x: listing)
        self.stub_out('os.path.exists', lambda x: True)
        self.stub_out('os.path.isfile', lambda x: True)

        base_dir = '/var/lib/nova/instances/_base'
        image_cache_manager = imagecache.ImageCacheManager()
        image_cache_manager._scan_base_images(base_dir)
        res = list(image_cache_manager._find_base_file(base_dir, fingerprint))

        base_file1 = os.path.join(base_dir, fingerprint)
        base_file2 = os.path.join(base_dir, fingerprint + '_sm')
        base_file3 = os.path.join(base_dir, fingerprint + '_10737418240')
        self.assertEqual(res, [(base_file1, False, False),
                               (base_file2, True, False),
                               (base_file3, False, True)])

    @contextlib.contextmanager
    def _make_base_file(self, lock=True, info=False):
        """Make a base file for testing."""

        with utils.tempdir() as tmpdir:
            self.flags(instances_path=tmpdir)
            self.flags(image_info_filename_pattern=('$instances_path/'
                                                    '%(image)s.info'),
                       group='libvirt')
            fname = os.path.join(tmpdir, 'aaa')

            base_file = open(fname, 'w')
            base_file.write('data')
            base_file.close()

            if lock:
                lockdir = os.path.join(tmpdir, 'locks')
                lockname = os.path.join(lockdir, 'nova-aaa')
                os.mkdir(lockdir)
                lock_file = open(lockname, 'w')
                lock_file.write('data')
                lock_file.close()

            base_file = open(fname, 'r')

            # TODO(mdbooth): Info files are no longer created by Newton,
            # but we must test that we continue to handle them correctly as
            # they may still be around from before the upgrade, and they may
            # be created by pre-Newton computes if we're on shared storage.
            # Once we can be sure that all computes are running at least
            # Newton (i.e. in Ocata), we can be sure that nothing is
            # creating info files any more, and we can delete the tests for
            # them.
            if info:
                # We're only checking for deletion, so contents are irrelevant
                open(imagecache.get_info_filename(fname), 'w').close()

            base_file.close()
            yield fname

    def test_remove_base_file(self):
        with self._make_base_file(info=True) as fname:
            image_cache_manager = imagecache.ImageCacheManager()
            image_cache_manager._remove_base_file(fname)
            info_fname = imagecache.get_info_filename(fname)

            lock_name = 'nova-' + os.path.split(fname)[-1]
            lock_dir = os.path.join(CONF.instances_path, 'locks')
            lock_file = os.path.join(lock_dir, lock_name)

            # Files are initially too new to delete
            self.assertTrue(os.path.exists(fname))
            self.assertTrue(os.path.exists(info_fname))
            self.assertTrue(os.path.exists(lock_file))

            # Old files get cleaned up though
            os.utime(fname, (-1, time.time() - 3601))
            image_cache_manager._remove_base_file(fname)

            self.assertFalse(os.path.exists(fname))
            self.assertFalse(os.path.exists(lock_file))

            # TODO(mdbooth): Remove test for deletion of info file in Ocata
            # (see comment in _make_base_file)
            self.assertFalse(os.path.exists(info_fname))

    def test_remove_base_file_original(self):
        with self._make_base_file(info=True) as fname:
            image_cache_manager = imagecache.ImageCacheManager()
            image_cache_manager.originals = [fname]
            image_cache_manager._remove_base_file(fname)
            info_fname = imagecache.get_info_filename(fname)

            # Files are initially too new to delete
            self.assertTrue(os.path.exists(fname))
            self.assertTrue(os.path.exists(info_fname))

            # This file should stay longer than a resized image
            os.utime(fname, (-1, time.time() - 3601))
            image_cache_manager._remove_base_file(fname)

            self.assertTrue(os.path.exists(fname))
            self.assertTrue(os.path.exists(info_fname))

            # Originals don't stay forever though
            os.utime(fname, (-1, time.time() - 3600 * 25))
            image_cache_manager._remove_base_file(fname)

            self.assertFalse(os.path.exists(fname))

            # TODO(mdbooth): Remove test for deletion of info file in Ocata
            # (see comment in _make_base_file)
            self.assertFalse(os.path.exists(info_fname))

    def test_remove_base_file_dne(self):
        # This test is solely to execute the "does not exist" code path. We
        # don't expect the method being tested to do anything in this case.
        with utils.tempdir() as tmpdir:
            self.flags(instances_path=tmpdir)
            self.flags(image_info_filename_pattern=('$instances_path/'
                                                    '%(image)s.info'),
                       group='libvirt')

            fname = os.path.join(tmpdir, 'aaa')
            image_cache_manager = imagecache.ImageCacheManager()
            image_cache_manager._remove_base_file(fname)

    def test_remove_base_file_oserror(self):
        with intercept_log_messages() as stream:
            with utils.tempdir() as tmpdir:
                self.flags(instances_path=tmpdir)
                self.flags(image_info_filename_pattern=('$instances_path/'
                                                        '%(image)s.info'),
                           group='libvirt')

                fname = os.path.join(tmpdir, 'aaa')

                os.mkdir(fname)
                os.utime(fname, (-1, time.time() - 3601))

                # This will raise an OSError because of file permissions
                image_cache_manager = imagecache.ImageCacheManager()
                image_cache_manager._remove_base_file(fname)

                self.assertTrue(os.path.exists(fname))
                self.assertNotEqual(stream.getvalue().find('Failed to remove'),
                                    -1)

    def test_handle_base_image_unused(self):
        img = '123'

        with self._make_base_file() as fname:
            os.utime(fname, (-1, time.time() - 3601))

            image_cache_manager = imagecache.ImageCacheManager()
            image_cache_manager.unexplained_images = [fname]
            image_cache_manager._handle_base_image(img, fname)

            self.assertEqual(image_cache_manager.unexplained_images, [])
            self.assertEqual(image_cache_manager.removable_base_files,
                             [fname])

    @mock.patch.object(libvirt_utils, 'update_mtime')
    def test_handle_base_image_used(self, mock_mtime):
        img = '123'

        with self._make_base_file() as fname:
            image_cache_manager = imagecache.ImageCacheManager()
            image_cache_manager.unexplained_images = [fname]
            image_cache_manager.used_images = {'123': (1, 0, ['banana-42'])}
            image_cache_manager._handle_base_image(img, fname)

            mock_mtime.assert_called_once_with(fname)
            self.assertEqual(image_cache_manager.unexplained_images, [])
            self.assertEqual(image_cache_manager.removable_base_files, [])

    @mock.patch.object(libvirt_utils, 'update_mtime')
    def test_handle_base_image_used_remotely(self, mock_mtime):
        img = '123'

        with self._make_base_file() as fname:
            image_cache_manager = imagecache.ImageCacheManager()
            image_cache_manager.unexplained_images = [fname]
            image_cache_manager.used_images = {'123': (0, 1, ['banana-42'])}
            image_cache_manager._handle_base_image(img, fname)

            mock_mtime.assert_called_once_with(fname)
            self.assertEqual(image_cache_manager.unexplained_images, [])
            self.assertEqual(image_cache_manager.removable_base_files, [])

    def test_handle_base_image_absent(self):
        img = '123'

        with intercept_log_messages() as stream:
            image_cache_manager = imagecache.ImageCacheManager()
            image_cache_manager.used_images = {'123': (1, 0, ['banana-42'])}
            image_cache_manager._handle_base_image(img, None)

            self.assertEqual(image_cache_manager.unexplained_images, [])
            self.assertEqual(image_cache_manager.removable_base_files, [])
            self.assertNotEqual(stream.getvalue().find('an absent base file'),
                                -1)

    def test_handle_base_image_used_missing(self):
        img = '123'

        with utils.tempdir() as tmpdir:
            self.flags(instances_path=tmpdir)
            self.flags(image_info_filename_pattern=('$instances_path/'
                                                    '%(image)s.info'),
                       group='libvirt')

            fname = os.path.join(tmpdir, 'aaa')

            image_cache_manager = imagecache.ImageCacheManager()
            image_cache_manager.unexplained_images = [fname]
            image_cache_manager.used_images = {'123': (1, 0, ['banana-42'])}
            image_cache_manager._handle_base_image(img, fname)

            self.assertEqual(image_cache_manager.unexplained_images, [])
            self.assertEqual(image_cache_manager.removable_base_files, [])

    @mock.patch.object(libvirt_utils, 'update_mtime')
    @mock.patch.object(lockutils, 'external_lock')
    def test_verify_base_images(self, mock_lock, mock_mtime):
        hashed_1 = '356a192b7913b04c54574d18c28d46e6395428ab'
        hashed_21 = '472b07b9fcf2c2451e8781e944bf5f77cd8457c8'
        hashed_22 = '12c6fc06c99a462375eeb3f43dfd832b08ca9e17'
        hashed_42 = '92cfceb39d57d914ed8b14d0e37643de0797ae56'

        self.flags(instances_path='/instance_path',
                   image_cache_subdirectory_name='_base')

        base_file_list = ['00000001',
                          'ephemeral_0_20_None',
                          'e97222e91fc4241f49a7f520d1dcf446751129b3_sm',
                          'e09c675c2d1cfac32dae3c2d83689c8c94bc693b_sm',
                          hashed_42,
                          hashed_1,
                          hashed_21,
                          hashed_22,
                          '%s_5368709120' % hashed_1,
                          '%s_10737418240' % hashed_1,
                          '00000004']

        def fq_path(path):
            return os.path.join('/instance_path/_base/', path)

        # Fake base directory existence
        orig_exists = os.path.exists

        def exists(path):
            # The python coverage tool got angry with my overly broad mocks
            if not path.startswith('/instance_path'):
                return orig_exists(path)

            if path in ['/instance_path',
                        '/instance_path/_base',
                        '/instance_path/instance-1/disk',
                        '/instance_path/instance-2/disk',
                        '/instance_path/instance-3/disk',
                        '/instance_path/_base/%s.info' % hashed_42]:
                return True

            for p in base_file_list:
                if path == fq_path(p):
                    return True
                if path == fq_path(p) + '.info':
                    return False

            if path in ['/instance_path/_base/%s_sm' % i for i in [hashed_1,
                                                                   hashed_21,
                                                                   hashed_22,
                                                                   hashed_42]]:
                return False

            self.fail('Unexpected path existence check: %s' % path)

        self.stub_out('os.path.exists', lambda x: exists(x))

        # Fake up some instances in the instances directory
        orig_listdir = os.listdir

        def listdir(path):
            # The python coverage tool got angry with my overly broad mocks
            if not path.startswith('/instance_path'):
                return orig_listdir(path)

            if path == '/instance_path':
                return ['instance-1', 'instance-2', 'instance-3', '_base']

            if path == '/instance_path/_base':
                return base_file_list

            self.fail('Unexpected directory listed: %s' % path)

        self.stub_out('os.listdir', lambda x: listdir(x))

        # Fake isfile for these faked images in _base
        orig_isfile = os.path.isfile

        def isfile(path):
            # The python coverage tool got angry with my overly broad mocks
            if not path.startswith('/instance_path'):
                return orig_isfile(path)

            for p in base_file_list:
                if path == fq_path(p):
                    return True

            self.fail('Unexpected isfile call: %s' % path)

        self.stub_out('os.path.isfile', lambda x: isfile(x))

        # Fake the database call which lists running instances
        instances = [{'image_ref': '1',
                      'host': CONF.host,
                      'name': 'instance-1',
                      'uuid': '123',
                      'vm_state': '',
                      'task_state': ''},
                     {'image_ref': '1',
                      'kernel_id': '21',
                      'ramdisk_id': '22',
                      'host': CONF.host,
                      'name': 'instance-2',
                      'uuid': '456',
                      'vm_state': '',
                      'task_state': ''}]
        all_instances = [fake_instance.fake_instance_obj(None, **instance)
                         for instance in instances]
        image_cache_manager = imagecache.ImageCacheManager()

        # Fake the utils call which finds the backing image
        def get_disk_backing_file(path):
            if path in ['/instance_path/instance-1/disk',
                        '/instance_path/instance-2/disk']:
                return fq_path('%s_5368709120' % hashed_1)
            self.fail('Unexpected backing file lookup: %s' % path)

        self.stubs.Set(libvirt_utils, 'get_disk_backing_file',
                       lambda x: get_disk_backing_file(x))

        # Fake getmtime as well
        orig_getmtime = os.path.getmtime

        def getmtime(path):
            if not path.startswith('/instance_path'):
                return orig_getmtime(path)

            return 1000000

        self.stub_out('os.path.getmtime', lambda x: getmtime(x))

        # Make sure we don't accidentally remove a real file
        orig_remove = os.remove

        def remove(path):
            if not path.startswith('/instance_path'):
                return orig_remove(path)

            # Don't try to remove fake files
            return

        self.stub_out('os.remove', lambda x: remove(x))

        self.mox.StubOutWithMock(objects.block_device.BlockDeviceMappingList,
                   'bdms_by_instance_uuid')

        ctxt = context.get_admin_context()
        objects.block_device.BlockDeviceMappingList.bdms_by_instance_uuid(
                ctxt, ['123', '456']).AndReturn({})

        self.mox.ReplayAll()
        # And finally we can make the call we're actually testing...
        # The argument here should be a context, but it is mocked out
        image_cache_manager.update(ctxt, all_instances)

        # Verify
        active = [fq_path(hashed_1), fq_path('%s_5368709120' % hashed_1),
                  fq_path(hashed_21), fq_path(hashed_22)]
        for act in active:
            self.assertIn(act, image_cache_manager.active_base_files)
        self.assertEqual(len(image_cache_manager.active_base_files),
                         len(active))

        for rem in [fq_path('e97222e91fc4241f49a7f520d1dcf446751129b3_sm'),
                    fq_path('e09c675c2d1cfac32dae3c2d83689c8c94bc693b_sm'),
                    fq_path(hashed_42),
                    fq_path('%s_10737418240' % hashed_1)]:
            self.assertIn(rem, image_cache_manager.removable_base_files)

    def test_verify_base_images_no_base(self):
        self.flags(instances_path='/tmp/no/such/dir/name/please')
        image_cache_manager = imagecache.ImageCacheManager()
        image_cache_manager.update(None, [])

    def test_is_valid_info_file(self):
        hashed = 'e97222e91fc4241f49a7f520d1dcf446751129b3'

        self.flags(instances_path='/tmp/no/such/dir/name/please')
        self.flags(image_info_filename_pattern=('$instances_path/_base/'
                                                '%(image)s.info'),
                   group='libvirt')
        base_filename = os.path.join(CONF.instances_path, '_base', hashed)

        is_valid_info_file = imagecache.is_valid_info_file
        self.assertFalse(is_valid_info_file('banana'))
        self.assertFalse(is_valid_info_file(
                os.path.join(CONF.instances_path, '_base', '00000001')))
        self.assertFalse(is_valid_info_file(base_filename))
        self.assertFalse(is_valid_info_file(base_filename + '.sha1'))
        self.assertTrue(is_valid_info_file(base_filename + '.info'))

    def test_run_image_cache_manager_pass(self):
        was = {'called': False}

        def fake_get_all_by_filters(context, *args, **kwargs):
            was['called'] = True
            instances = []
            for x in range(2):
                instances.append(fake_instance.fake_db_instance(
                                                        image_ref='1',
                                                        uuid=x,
                                                        name=x,
                                                        vm_state='',
                                                        task_state=''))
            return instances

        with utils.tempdir() as tmpdir:
            self.flags(instances_path=tmpdir)

            self.stub_out('nova.db.instance_get_all_by_filters',
                          fake_get_all_by_filters)
            compute = importutils.import_object(CONF.compute_manager)
            self.flags(use_local=True, group='conductor')
            compute.conductor_api = conductor.API()
            ctxt = context.get_admin_context()
            compute._run_image_cache_manager_pass(ctxt)
            self.assertTrue(was['called'])

    def test_store_swap_image(self):
        image_cache_manager = imagecache.ImageCacheManager()
        image_cache_manager._store_swap_image('swap_')
        image_cache_manager._store_swap_image('swap_123')
        image_cache_manager._store_swap_image('swap_456')
        image_cache_manager._store_swap_image('swap_abc')
        image_cache_manager._store_swap_image('123_swap')
        image_cache_manager._store_swap_image('swap_129_')

        self.assertEqual(len(image_cache_manager.back_swap_images), 2)
        expect_set = set(['swap_123', 'swap_456'])
        self.assertEqual(image_cache_manager.back_swap_images, expect_set)

    @mock.patch.object(lockutils, 'external_lock')
    @mock.patch.object(libvirt_utils, 'update_mtime')
    @mock.patch('os.path.exists')
    @mock.patch('os.path.getmtime')
    @mock.patch('os.remove')
    def test_age_and_verify_swap_images(self, mock_remove, mock_getmtime,
            mock_exist, mock_mtime, mock_lock):
        base_dir = '/tmp_age_test'
        self.flags(image_info_filename_pattern=base_dir + '/%(image)s.info',
                   group='libvirt')

        image_cache_manager = imagecache.ImageCacheManager()
        expected_remove = set()
        expected_exist = set(['swap_128', 'swap_256'])

        image_cache_manager.back_swap_images.add('swap_128')
        image_cache_manager.back_swap_images.add('swap_256')

        image_cache_manager.used_swap_images.add('swap_128')

        mock_getmtime.side_effect = lambda path: time.time() - 1000000

        mock_exist.side_effect = \
            lambda path: os.path.dirname(path) == base_dir and \
                         os.path.basename(path) in expected_exist

        def removefile(path):
            self.assertEqual(base_dir, os.path.dirname(path),
                             'Attempt to remove unexpected path')

            fn = os.path.basename(path)
            expected_remove.add(fn)
            expected_exist.remove(fn)

        mock_remove.side_effect = removefile

        image_cache_manager._age_and_verify_swap_images(None, base_dir)
        self.assertEqual(set(['swap_128']), expected_exist)
        self.assertEqual(set(['swap_256']), expected_remove)

    @mock.patch.object(utils, 'synchronized')
    @mock.patch.object(imagecache.ImageCacheManager, '_get_age_of_file',
                       return_value=(True, 100))
    def test_lock_acquired_on_removing_old_enough_files(self, mock_get_age,
                                                        mock_synchronized):
        base_file = '/tmp_age_test'
        lock_path = os.path.join(CONF.instances_path, 'locks')
        lock_file = os.path.split(base_file)[-1]
        image_cache_manager = imagecache.ImageCacheManager()
        image_cache_manager._remove_old_enough_file(base_file, 60,
                                                    remove_lock=False)
        mock_synchronized.assert_called_once_with(lock_file, external=True,
                                                  lock_path=lock_path)


class ImageCacheLocalDirTestCase(test.NoDBTestCase):
    def setUp(self):
        super(ImageCacheLocalDirTestCase, self).setUp()

        # Ensure we get a new singleton with each test run
        imagecache.ImageCacheLocalDir._singleton = None

        self.image_id = 'e557355c-19e9-4749-837a-87a222610fea'
        self.image_hash = '87ee9ef6f36612843bc8e61eb4ab6a1593380891'

        self.instances_path = '/instances'
        self.default_lock_path = os.path.join(self.instances_path, 'locks')
        self.default_cache_dir = os.path.join(self.instances_path, '_base')

        self.expected_image_path = os.path.join(self.default_cache_dir,
                                                self.image_hash)

        self.func_name = 'swap_128'
        self.expected_func_path = os.path.join(self.default_cache_dir,
                                               self.func_name)

        self.flags(instances_path=self.instances_path)

        self.locked = False

        # True if the default cache dir exists. We needn't be locked to test
        # this.
        self.cache_dir_exists = True
        # The set of paths which exist. We must be locked to test these.
        self.existing_paths = set()

    def fake_synchronized(self, fn):
        # A fake synchronized method which toggles the locked property while
        # held. We can check this field in tests to assert we are holding
        # the lock during operations.
        @six.wraps(fn)
        def wrapper(*args, **kwargs):
            self.locked = True
            fn(*args, **kwargs)
            self.locked = False

        return wrapper

    def assert_in_lock(self, *args, **kwargs):
        self.assertTrue(self.locked)
        return mock.DEFAULT

    def fake_exists(self, path):
        # Don't need to be locked when testing the cache directory
        if path == self.default_cache_dir:
            return self.cache_dir_exists

        # Should be locked when testing anything else
        self.assert_in_lock()
        return path in self.existing_paths

    @contextlib.contextmanager
    def default_mocks(self):
        with mock.patch.object(utils, 'synchronized', autospec=True,
                               return_value=self.fake_synchronized) \
                as self.mock_synchronized, \
                mock.patch.object(os.path, 'exists', autospec=True,
                                  side_effect=self.fake_exists) \
                        as self.mock_exists, \
                mock.patch.object(images, 'qemu_img_info', autospec=True) \
                        as self.mock_qemu_img_info:
            yield

    def assert_qemu_img_info(self, result):
        # Check that result corresponds to the return of qemu_img_info
        self.mock_qemu_img_info.assert_called_once_with(
            self.expected_image_path)
        self.assertEqual(self.expected_image_path, result.path)
        self.assertIs(self.mock_qemu_img_info.return_value.file_format,
                      result.file_format)
        self.assertIs(self.mock_qemu_img_info.return_value.virtual_size,
                      result.virtual_size)
        self.assertIs(self.mock_qemu_img_info.return_value.disk_size,
                      result.disk_size)

    def test_singleton(self):
        # Multiple calls to ImageCacheLocalDir.get() should return the same
        # object
        with self.default_mocks():
            cache1 = imagecache.ImageCacheLocalDir.get()
            cache2 = imagecache.ImageCacheLocalDir.get()

        self.assertIs(cache1, cache2)

    @mock.patch.object(fileutils, 'ensure_tree', autospec=True)
    def test_create_cache_dir(self, mock_ensure_tree):
        # Creating a pool should create the cache directory if it doesn't exist
        self.cache_dir_exists = False

        with self.default_mocks():
            imagecache.ImageCacheLocalDir.get()

        mock_ensure_tree.assert_called_once_with(self.default_cache_dir)

    @mock.patch.object(libvirt_utils, 'update_mtime', autospec=True)
    def test_get_image_info_exists(self, mock_update_mtime):
        # Calling get_image_info when the image exists should return the
        # existing image without fetching it.

        # Image path already exists
        self.existing_paths.add(self.expected_image_path)

        with self.default_mocks():
            pool = imagecache.ImageCacheLocalDir.get()

            result = pool.get_image_info(mock.sentinel.context, self.image_id)
        self.assert_qemu_img_info(result)

        mock_update_mtime.assert_called_once_with(self.expected_image_path)

        self.mock_synchronized.assert_called_once_with(
            self.image_hash, external=True,
            lock_path=self.default_lock_path)

    @mock.patch.object(libvirt_utils, 'fetch_image', autospec=True)
    def test_get_image_info_not_exists(self, mock_fetch_image):
        # Calling get_image_info when the image does not exist locally should
        # fetch the image before returning it.

        # Assert that we're locked when fetching the image
        mock_fetch_image.side_effect = self.assert_in_lock

        with self.default_mocks():
            pool = imagecache.ImageCacheLocalDir.get()

            result = pool.get_image_info(mock.sentinel.context, self.image_id)
        self.assert_qemu_img_info(result)

        self.mock_synchronized.assert_called_once_with(
            self.image_hash, external=True,
            lock_path=self.default_lock_path)

        mock_fetch_image.assert_called_once_with(
            mock.sentinel.context, self.expected_image_path, self.image_id)

    @mock.patch.object(libvirt_utils, 'fetch_image', autospec=True)
    def test_get_image_info_not_exists_no_fallback(self, mock_fetch_image):
        # Calling get_image_info when the image does not exist
        # locally, it is not available from glance, and no fallback is given
        # should raise ImageNotFound
        mock_fetch_image.side_effect = exception.ImageNotFound(
            image_id=self.image_id)

        with self.default_mocks():
            pool = imagecache.ImageCacheLocalDir.get()

            self.assertRaises(exception.ImageNotFound, pool.get_image_info,
                              mock.sentinel.context, self.image_id)

        self.mock_synchronized.assert_called_once_with(
            self.image_hash, external=True,
            lock_path=self.default_lock_path)

        mock_fetch_image.assert_called_once_with(
            mock.sentinel.context, self.expected_image_path, self.image_id)

    @mock.patch.object(libvirt_utils, 'copy_image', autospec=True)
    @mock.patch.object(libvirt_utils, 'fetch_image', autospec=True)
    def test_get_image_info_not_exists_fallback(self, mock_fetch_image,
                                                  mock_copy_image):
        # Calling get_image_info when the image does not exist
        # locally and it is not available from glance should fallback to
        # copying from a remote host if one was given.
        mock_fetch_image.side_effect = exception.ImageNotFound(
            image_id=self.image_id)

        with self.default_mocks():
            pool = imagecache.ImageCacheLocalDir.get()

            result = pool.get_image_info(mock.sentinel.context, self.image_id,
                                         fallback='host')
        self.assert_qemu_img_info(result)

        self.mock_synchronized.assert_called_once_with(
            self.image_hash, external=True,
            lock_path=self.default_lock_path)

        mock_fetch_image.assert_called_once_with(
            mock.sentinel.context, self.expected_image_path, self.image_id)
        mock_copy_image.assert_called_once_with(src=self.expected_image_path,
                                                dest=self.expected_image_path,
                                                host='host',
                                                receive=True)

    @mock.patch.object(libvirt_utils, 'update_mtime', autospec=True)
    def test_get_func_output_path_exists(self, mock_update_mtime):
        # Calling get_func_output_path when the file exists should update its
        # mtime and not raise any exceptions
        fallback_host = uuidutils.generate_uuid()
        self.existing_paths.add(self.expected_func_path)

        mock_create = mock.Mock()

        with self.default_mocks():
            pool = imagecache.ImageCacheLocalDir.get()
            pool.get_func_output_path(mock_create, self.func_name,
                                      fallback=fallback_host)

        self.mock_exists.assert_any_call(self.expected_func_path)
        mock_update_mtime.assert_called_once_with(self.expected_func_path)
        self.assertFalse(mock_create.called)

    @mock.patch.object(libvirt_utils, 'copy_image', autospec=True)
    def test_get_func_output_path_not_exists(self, mock_copy_image):
        # Calling get_func_output_path when the file does not exist should
        # attempt to retrieve it from the fallback host

        # We should be locked while copying the image
        mock_copy_image.side_effect = \
            lambda *args, **kwargs: self.assertTrue(self.locked)

        mock_create = mock.Mock()

        with self.default_mocks():
            pool = imagecache.ImageCacheLocalDir.get()
            pool.get_func_output_path(mock_create, self.func_name,
                                      fallback=mock.sentinel.fallback)

        self.mock_exists.assert_any_call(self.expected_func_path)
        mock_copy_image.assert_called_once_with(src=self.expected_func_path,
                                                dest=self.expected_func_path,
                                                host=mock.sentinel.fallback,
                                                receive=True)
        self.assertFalse(mock_create.called)

    @mock.patch.object(libvirt_utils, 'copy_image', autospec=True)
    def test_get_func_output_path_not_exists_no_fallback(self,
                                                         mock_copy_image):
        # Calling get_func_output_path when the file does not exist should
        # attempt to retrieve it from the fallback host

        mock_copy_image.side_effect = processutils.ProcessExecutionError
        mock_create = mock.Mock()

        with self.default_mocks():
            cache = imagecache.ImageCacheLocalDir.get()
            cache.get_func_output_path(mock_create, self.func_name,
                                       fallback=mock.sentinel.fallback)

        self.mock_exists.assert_any_call(self.expected_func_path)
        mock_copy_image.assert_called_once_with(src=self.expected_func_path,
                                                dest=self.expected_func_path,
                                                host=mock.sentinel.fallback,
                                                receive=True)
        mock_create.assert_called_once_with(self.expected_func_path)
