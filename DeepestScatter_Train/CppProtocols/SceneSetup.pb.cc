// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: SceneSetup.proto

#include "SceneSetup.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// This is a temporary google only hack
#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
#include "third_party/protobuf/version.h"
#endif
// @@protoc_insertion_point(includes)

namespace protobuf_Vector_2eproto {
extern PROTOBUF_INTERNAL_EXPORT_protobuf_Vector_2eproto ::google::protobuf::internal::SCCInfo<0> scc_info_Vector3;
}  // namespace protobuf_Vector_2eproto
namespace Persistance {
class SceneSetupDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<SceneSetup>
      _instance;
} _SceneSetup_default_instance_;
}  // namespace Persistance
namespace protobuf_SceneSetup_2eproto {
static void InitDefaultsSceneSetup() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::Persistance::_SceneSetup_default_instance_;
    new (ptr) ::Persistance::SceneSetup();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::Persistance::SceneSetup::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<1> scc_info_SceneSetup =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 1, InitDefaultsSceneSetup}, {
      &protobuf_Vector_2eproto::scc_info_Vector3.base,}};

void InitDefaults() {
  ::google::protobuf::internal::InitSCC(&scc_info_SceneSetup.base);
}

::google::protobuf::Metadata file_level_metadata[1];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::Persistance::SceneSetup, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::Persistance::SceneSetup, cloud_path_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::Persistance::SceneSetup, cloud_size_m_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::Persistance::SceneSetup, light_direction_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::Persistance::SceneSetup)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::Persistance::_SceneSetup_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "SceneSetup.proto", schemas, file_default_instances, TableStruct::offsets,
      file_level_metadata, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_PROTOBUF_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 1);
}

void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n\020SceneSetup.proto\022\013Persistance\032\014Vector."
      "proto\"e\n\nSceneSetup\022\022\n\ncloud_path\030\001 \001(\t\022"
      "\024\n\014cloud_size_m\030\002 \001(\002\022-\n\017light_direction"
      "\030\003 \001(\0132\024.Persistance.Vector3b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 156);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "SceneSetup.proto", &protobuf_RegisterTypes);
  ::protobuf_Vector_2eproto::AddDescriptors();
}

void AddDescriptors() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at dynamic initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;
}  // namespace protobuf_SceneSetup_2eproto
namespace Persistance {

// ===================================================================

void SceneSetup::InitAsDefaultInstance() {
  ::Persistance::_SceneSetup_default_instance_._instance.get_mutable()->light_direction_ = const_cast< ::Persistance::Vector3*>(
      ::Persistance::Vector3::internal_default_instance());
}
void SceneSetup::clear_light_direction() {
  if (GetArenaNoVirtual() == NULL && light_direction_ != NULL) {
    delete light_direction_;
  }
  light_direction_ = NULL;
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int SceneSetup::kCloudPathFieldNumber;
const int SceneSetup::kCloudSizeMFieldNumber;
const int SceneSetup::kLightDirectionFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

SceneSetup::SceneSetup()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_SceneSetup_2eproto::scc_info_SceneSetup.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:Persistance.SceneSetup)
}
SceneSetup::SceneSetup(const SceneSetup& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  cloud_path_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.cloud_path().size() > 0) {
    cloud_path_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.cloud_path_);
  }
  if (from.has_light_direction()) {
    light_direction_ = new ::Persistance::Vector3(*from.light_direction_);
  } else {
    light_direction_ = NULL;
  }
  cloud_size_m_ = from.cloud_size_m_;
  // @@protoc_insertion_point(copy_constructor:Persistance.SceneSetup)
}

void SceneSetup::SharedCtor() {
  cloud_path_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  ::memset(&light_direction_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&cloud_size_m_) -
      reinterpret_cast<char*>(&light_direction_)) + sizeof(cloud_size_m_));
}

SceneSetup::~SceneSetup() {
  // @@protoc_insertion_point(destructor:Persistance.SceneSetup)
  SharedDtor();
}

void SceneSetup::SharedDtor() {
  cloud_path_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (this != internal_default_instance()) delete light_direction_;
}

void SceneSetup::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* SceneSetup::descriptor() {
  ::protobuf_SceneSetup_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_SceneSetup_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const SceneSetup& SceneSetup::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_SceneSetup_2eproto::scc_info_SceneSetup.base);
  return *internal_default_instance();
}


void SceneSetup::Clear() {
// @@protoc_insertion_point(message_clear_start:Persistance.SceneSetup)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cloud_path_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (GetArenaNoVirtual() == NULL && light_direction_ != NULL) {
    delete light_direction_;
  }
  light_direction_ = NULL;
  cloud_size_m_ = 0;
  _internal_metadata_.Clear();
}

bool SceneSetup::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:Persistance.SceneSetup)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // string cloud_path = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_cloud_path()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->cloud_path().data(), static_cast<int>(this->cloud_path().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "Persistance.SceneSetup.cloud_path"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // float cloud_size_m = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(21u /* 21 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &cloud_size_m_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // .Persistance.Vector3 light_direction = 3;
      case 3: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(26u /* 26 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessage(
               input, mutable_light_direction()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:Persistance.SceneSetup)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:Persistance.SceneSetup)
  return false;
#undef DO_
}

void SceneSetup::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:Persistance.SceneSetup)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string cloud_path = 1;
  if (this->cloud_path().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->cloud_path().data(), static_cast<int>(this->cloud_path().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "Persistance.SceneSetup.cloud_path");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      1, this->cloud_path(), output);
  }

  // float cloud_size_m = 2;
  if (this->cloud_size_m() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(2, this->cloud_size_m(), output);
  }

  // .Persistance.Vector3 light_direction = 3;
  if (this->has_light_direction()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      3, this->_internal_light_direction(), output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:Persistance.SceneSetup)
}

::google::protobuf::uint8* SceneSetup::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:Persistance.SceneSetup)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string cloud_path = 1;
  if (this->cloud_path().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->cloud_path().data(), static_cast<int>(this->cloud_path().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "Persistance.SceneSetup.cloud_path");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        1, this->cloud_path(), target);
  }

  // float cloud_size_m = 2;
  if (this->cloud_size_m() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(2, this->cloud_size_m(), target);
  }

  // .Persistance.Vector3 light_direction = 3;
  if (this->has_light_direction()) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageToArray(
        3, this->_internal_light_direction(), deterministic, target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:Persistance.SceneSetup)
  return target;
}

size_t SceneSetup::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:Persistance.SceneSetup)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // string cloud_path = 1;
  if (this->cloud_path().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->cloud_path());
  }

  // .Persistance.Vector3 light_direction = 3;
  if (this->has_light_direction()) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::MessageSize(
        *light_direction_);
  }

  // float cloud_size_m = 2;
  if (this->cloud_size_m() != 0) {
    total_size += 1 + 4;
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void SceneSetup::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:Persistance.SceneSetup)
  GOOGLE_DCHECK_NE(&from, this);
  const SceneSetup* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const SceneSetup>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:Persistance.SceneSetup)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:Persistance.SceneSetup)
    MergeFrom(*source);
  }
}

void SceneSetup::MergeFrom(const SceneSetup& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:Persistance.SceneSetup)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.cloud_path().size() > 0) {

    cloud_path_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.cloud_path_);
  }
  if (from.has_light_direction()) {
    mutable_light_direction()->::Persistance::Vector3::MergeFrom(from.light_direction());
  }
  if (from.cloud_size_m() != 0) {
    set_cloud_size_m(from.cloud_size_m());
  }
}

void SceneSetup::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:Persistance.SceneSetup)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void SceneSetup::CopyFrom(const SceneSetup& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:Persistance.SceneSetup)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SceneSetup::IsInitialized() const {
  return true;
}

void SceneSetup::Swap(SceneSetup* other) {
  if (other == this) return;
  InternalSwap(other);
}
void SceneSetup::InternalSwap(SceneSetup* other) {
  using std::swap;
  cloud_path_.Swap(&other->cloud_path_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  swap(light_direction_, other->light_direction_);
  swap(cloud_size_m_, other->cloud_size_m_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata SceneSetup::GetMetadata() const {
  protobuf_SceneSetup_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_SceneSetup_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace Persistance
namespace google {
namespace protobuf {
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::Persistance::SceneSetup* Arena::CreateMaybeMessage< ::Persistance::SceneSetup >(Arena* arena) {
  return Arena::CreateInternal< ::Persistance::SceneSetup >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
