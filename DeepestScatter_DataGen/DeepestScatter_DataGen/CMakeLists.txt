project(DeepestScatter_DataGen CXX)

################################################################################
# Source groups
################################################################################
set(src
    installers.cpp
    src/installers.h
    src/main.cpp
    src/Mie.cpp
    src/Mie.h
)
source_group("src" FILES ${src})

set(src__CUDA
    src/CUDA/cloud.cuh
    src/CUDA/DisneyDescriptor.cuh
    src/CUDA/DisneyDescriptor.h
    src/CUDA/optixExtraMath.cuh
    src/CUDA/PointRadianceTask.h
    src/CUDA/random.cuh
    src/CUDA/rayData.cuh
)
source_group("src\\CUDA" FILES ${src__CUDA})

set(src__ExecutionLoop
    src/ExecutionLoop/ExecutionLoop.h
    src/ExecutionLoop/GuiExecutionLoop.cpp
    src/ExecutionLoop/GuiExecutionLoop.h
    src/ExecutionLoop/Tasks.cpp
    src/ExecutionLoop/Tasks.h
)
source_group("src\\ExecutionLoop" FILES ${src__ExecutionLoop})

set(src__Hypodermic
    src/Hypodermic/ActivatedRegistrationInfo.h
    src/Hypodermic/ActivationException.h
    src/Hypodermic/ActivationHandler.h
    src/Hypodermic/ActivationHandlers.h
    src/Hypodermic/ActivationRegistry.h
    src/Hypodermic/ActivationResult.h
    src/Hypodermic/AnyArgument.h
    src/Hypodermic/ArgumentPack.h
    src/Hypodermic/ArgumentResolver.h
    src/Hypodermic/As.h
    src/Hypodermic/AsSelf.h
    src/Hypodermic/AutowireableConstructor.h
    src/Hypodermic/AutowireableConstructorRegistrationDescriptor.h
    src/Hypodermic/Behavior.h
    src/Hypodermic/CircularDependencyException.h
    src/Hypodermic/ComponentContext.h
    src/Hypodermic/Config.h
    src/Hypodermic/ConsoleLoggerSink.h
    src/Hypodermic/ConstructorDescriptor.h
    src/Hypodermic/ConstructorTypologyDeducer.h
    src/Hypodermic/Container.h
    src/Hypodermic/ContainerBuilder.h
    src/Hypodermic/ContainerInstanceRegistration.h
    src/Hypodermic/ContainerInstanceRegistrationActivator.h
    src/Hypodermic/DependencyActivationException.h
    src/Hypodermic/DependencyFactories.h
    src/Hypodermic/DependencyFactory.h
    src/Hypodermic/DependencyFactoryTag.h
    src/Hypodermic/ExceptionBase.h
    src/Hypodermic/FactoryWrapper.h
    src/Hypodermic/Hypodermic.h
    src/Hypodermic/Hypodermic.vcxproj
    src/Hypodermic/Hypodermic.vcxproj.filters
    src/Hypodermic/ILoggerSink.h
    src/Hypodermic/InstanceAlreadyActivatingException.h
    src/Hypodermic/InstanceFactory.h
    src/Hypodermic/InstanceLifetime.h
    src/Hypodermic/InstanceRegistrationTags.h
    src/Hypodermic/IntegerSequence.h
    src/Hypodermic/IRegistration.h
    src/Hypodermic/IRegistrationActivator.h
    src/Hypodermic/IRegistrationDescriptor.h
    src/Hypodermic/IRegistrationRegistry.h
    src/Hypodermic/IRegistrationScope.h
    src/Hypodermic/IResolutionContainer.h
    src/Hypodermic/IResolutionContext.h
    src/Hypodermic/IRuntimeRegistrationBuilder.h
    src/Hypodermic/IsComplete.h
    src/Hypodermic/IsSupportedArgument.h
    src/Hypodermic/ITypeAlias.h
    src/Hypodermic/libhypodermic.pc.in
    src/Hypodermic/Log.h
    src/Hypodermic/Logger.h
    src/Hypodermic/LogLevel.h
    src/Hypodermic/MetaContains.h
    src/Hypodermic/MetaForEach.h
    src/Hypodermic/MetaIdentity.h
    src/Hypodermic/MetaInsert.h
    src/Hypodermic/MetaMap.h
    src/Hypodermic/MetaPair.h
    src/Hypodermic/Named.h
    src/Hypodermic/NamedTypeAlias.h
    src/Hypodermic/NestedRegistrationScope.h
    src/Hypodermic/NoopLoggerSink.h
    src/Hypodermic/OnActivated.h
    src/Hypodermic/PersistentInstanceRegistration.h
    src/Hypodermic/PointerAlignment.h
    src/Hypodermic/Pragmas.h
    src/Hypodermic/ProvidedDependencyTag.h
    src/Hypodermic/ProvidedInstanceDependencyTag.h
    src/Hypodermic/ProvidedInstanceFactoryRegistrationDescriptor.h
    src/Hypodermic/ProvidedInstanceRegistration.h
    src/Hypodermic/ProvidedInstanceRegistrationActivator.h
    src/Hypodermic/ProvidedInstanceRegistrationDescriptor.h
    src/Hypodermic/Registration.h
    src/Hypodermic/RegistrationActivator.h
    src/Hypodermic/RegistrationBuilder.h
    src/Hypodermic/RegistrationContext.h
    src/Hypodermic/RegistrationDescriptorBase.h
    src/Hypodermic/RegistrationDescriptorBuilder.h
    src/Hypodermic/RegistrationDescriptorInfo.h
    src/Hypodermic/RegistrationDescriptorInfoToString.h
    src/Hypodermic/RegistrationException.h
    src/Hypodermic/RegistrationScope.h
    src/Hypodermic/ResolutionContainer.h
    src/Hypodermic/ResolutionContext.h
    src/Hypodermic/ResolutionException.h
    src/Hypodermic/ResolutionInfo.h
    src/Hypodermic/RuntimeRegistrationBuilder.h
    src/Hypodermic/SingleInstance.h
    src/Hypodermic/TypeAlias.h
    src/Hypodermic/TypeAliases.h
    src/Hypodermic/TypeAliasKey.h
    src/Hypodermic/TypeInfo.h
    src/Hypodermic/UseIfNone.h
    src/Hypodermic/With.h
)
source_group("src\\Hypodermic" FILES ${src__Hypodermic})

set(src__Model__Disney
    src/Model/Disney/Predictor.cpp
    src/Model/Disney/Predictor.h
)
source_group("src\\Model\\Disney" FILES ${src__Model__Disney})

set(src__Scene
    src/Scene/Camera.cpp
    src/Scene/Camera.h
    src/Scene/CloudPTRenderer.cpp
    src/Scene/CloudPTRenderer.h
    src/Scene/DisneyDescriptorCollector.cpp
    src/Scene/DisneyDescriptorCollector.h
    src/Scene/RadianceCollector.cpp
    src/Scene/RadianceCollector.h
    src/Scene/ScatterSampleCollector.cpp
    src/Scene/ScatterSampleCollector.h
    src/Scene/Scene.cpp
    src/Scene/Scene.h
    src/Scene/SceneDescription.h
    src/Scene/SceneItem.h
    src/Scene/Sun.cpp
    src/Scene/Sun.h
    src/Scene/VDBCloud.cpp
    src/Scene/VDBCloud.h
)
source_group("src\\Scene" FILES ${src__Scene})

set(src__Uitl
    src/Util/Arcball.cpp
    src/Util/Arcball.h
    src/Util/BufferBind.h
    src/Util/Resources.cpp
    src/Util/Resources.h
    src/Util/sutil.cpp
    src/Util/sutil.h
    src/Util/sutilapi.h
)
source_group("src\\Uitl" FILES ${src__Uitl})

set(src__Uitl__Dataset
    src/Util/Dataset/BatchSettings.h
    src/Util/Dataset/Dataset.cpp
    src/Util/Dataset/Dataset.h
    src/Util/Dataset/LmdbExceptions.cpp
    src/Util/Dataset/LmdbExceptions.h
    src/Util/Dataset/Transaction.cpp
    src/Util/Dataset/Transaction.h
)
source_group("src\\Uitl\\Dataset" FILES ${src__Uitl__Dataset})

set(ALL_FILES  ${src} ${src__ExecutionLoop} ${src__Model__Disney} ${src__Scene} ${src__Uitl} ${src__Uitl__Dataset} ${src__CUDA} ${src__Hypodermic})

################################################################################
# Target
################################################################################
add_executable(${PROJECT_NAME} ${ALL_FILES})

if(NOT ("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64"))
    message(WARNING "${CMAKE_VS_PLATFORM_NAME} arch is not supported!")
endif()

################################################################################
# Target name
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    string(CONCAT TARGET_NAME
          $<$<CONFIG:Debug>:${PROJECT_NAME}>
        $<$<CONFIG:Release>:${PROJECT_NAME}>
        $<$<NOT:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>>>:${PROJECT_NAME}>
    )
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${TARGET_NAME})

################################################################################
# Output directory
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    string(CONCAT OUT_DIR
          $<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}//$<CONFIG>/>
        $<$<CONFIG:Release>:${CMAKE_SOURCE_DIR}//$<CONFIG>/>
        $<$<NOT:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>>>:${CMAKE_SOURCE_DIR}/${CMAKE_VS_PLATFORM_NAME}/$<CONFIG>>
    )
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})

################################################################################
# Include directories
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_include_directories(${PROJECT_NAME} PUBLIC
          $<$<CONFIG:Debug>:${CMAKE_CURRENT_SOURCE_DIR}/../../DeepestScatter_Train/CppProtocols;${CMAKE_CURRENT_SOURCE_DIR}/src;$ENV{ILMBASE_ROOT}/include;$ENV{TBB_ROOT}/include;$ENV{OPENEXR_ROOT}/include;$ENV{BOOST_ROOT};$ENV{OPENVDB_ROOT}/include;$ENV{CUDA_PATH}/include;$ENV{OPTIX_SDK_PATH}/include>
        $<$<CONFIG:Release>:${CMAKE_CURRENT_SOURCE_DIR}/../../DeepestScatter_Train/CppProtocols;${CMAKE_CURRENT_SOURCE_DIR}/src;$ENV{ILMBASE_ROOT}/include;$ENV{TBB_ROOT}/include;$ENV{OPENEXR_ROOT}/include;$ENV{BOOST_ROOT};$ENV{OPENVDB_ROOT}/include;$ENV{CUDA_PATH}/include;$ENV{OPTIX_SDK_PATH}/include>
    )
endif()

################################################################################
# Compile definitions
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
          $<$<CONFIG:Debug>:BOOST_DI_CFG_DIAGNOSTICS_LEVEL=2;OPENVDB_ABI_VERSION_NUMBER=5;OPENEXR_DLL;NOMINMAX;WIN64;_DEBUG;_CONSOLE;_MBCS>
        $<$<CONFIG:Release>:BOOST_DI_CFG_DIAGNOSTICS_LEVEL=2;OPENVDB_ABI_VERSION_NUMBER=5;OPENEXR_DLL;NOMINMAX;WIN64;NDEBUG;_CONSOLE;_MBCS>
    )
endif()

################################################################################
# Compile and link options
################################################################################
if(MSVC)
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_compile_options(${PROJECT_NAME} PRIVATE
              $<$<CONFIG:Debug>:/Od;/MDd;/W3;/fp:strict;/GR;/bigobj;/EHsc>
            $<$<CONFIG:Release>:/GL;/O2;/Oi;/MD;/Gy;/W3;/fp:strict;/GR;/bigobj;/EHsc>
        )
    endif()
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS_DEBUG "/DEBUG /INCREMENTAL /wholearchive:caffe2_CPU")
        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS_RELEASE "/LTCG /DEBUG /wholearchive:caffe2_CPU")
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS_DEBUG "/DEBUG /INCREMENTAL /wholearchive:caffe2_CPU")
        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS_RELEASE "/LTCG /DEBUG /wholearchive:caffe2_CPU")
    endif()
endif()

################################################################################
# Post build events
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    add_custom_command_if(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMANDS
        COMMAND   $<CONFIG:Debug> echo copy "$ENV{CudaToolkitBinDir}\\cudart*.dll" "$<SHELL_PATH:${OUT_DIR}>"
        COMMAND   $<CONFIG:Debug> copy "$ENV{CudaToolkitBinDir}\\cudart*.dll" "$<SHELL_PATH:${OUT_DIR}>"
        COMMAND $<CONFIG:Release> echo copy "$ENV{CudaToolkitBinDir}\\cudart*.dll" "$<SHELL_PATH:${OUT_DIR}>"
        COMMAND $<CONFIG:Release> copy "$ENV{CudaToolkitBinDir}\\cudart*.dll" "$<SHELL_PATH:${OUT_DIR}>"
    )
endif()

################################################################################
# Dependencies
################################################################################
add_dependencies(${PROJECT_NAME} CppProtocols)

# Link with other additional libraries.
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_link_libraries(${PROJECT_NAME} PUBLIC
          $<$<CONFIG:Debug>:Caffe2_CPU$<SEMICOLON>glog$<SEMICOLON>CppProtocols$<SEMICOLON>Half$<SEMICOLON>gflagsd$<SEMICOLON>lmdbd$<SEMICOLON>tbb_debug$<SEMICOLON>openvdb$<SEMICOLON>$(OPTIX_SDK_PATH)\lib64\optix.51$<SEMICOLON>$(OPTIX_SDK_PATH)\lib64\optixu.1$<SEMICOLON>$(CUDA_PATH)\lib\x64\nvrtc$<SEMICOLON>winmm$<SEMICOLON>cudart$<SEMICOLON>kernel32$<SEMICOLON>user32$<SEMICOLON>gdi32$<SEMICOLON>winspool$<SEMICOLON>comdlg32$<SEMICOLON>advapi32$<SEMICOLON>shell32$<SEMICOLON>ole32$<SEMICOLON>oleaut32$<SEMICOLON>uuid$<SEMICOLON>odbc32$<SEMICOLON>odbccp32>
        $<$<CONFIG:Release>:Caffe2_CPU$<SEMICOLON>glog$<SEMICOLON>CppProtocols$<SEMICOLON>Half$<SEMICOLON>gflags$<SEMICOLON>lmdb$<SEMICOLON>tbb$<SEMICOLON>openvdb$<SEMICOLON>$(OPTIX_SDK_PATH)\lib64\optix.51$<SEMICOLON>$(OPTIX_SDK_PATH)\lib64\optixu.1$<SEMICOLON>$(CUDA_PATH)\lib\x64\nvrtc$<SEMICOLON>winmm$<SEMICOLON>cudart$<SEMICOLON>kernel32$<SEMICOLON>user32$<SEMICOLON>gdi32$<SEMICOLON>winspool$<SEMICOLON>comdlg32$<SEMICOLON>advapi32$<SEMICOLON>shell32$<SEMICOLON>ole32$<SEMICOLON>oleaut32$<SEMICOLON>uuid$<SEMICOLON>odbc32$<SEMICOLON>odbccp32>
    )
endif()
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_link_directories(${PROJECT_NAME} PUBLIC
        CONDITION   $<CONFIG:Debug> "${CMAKE_SOURCE_DIR}/$<CONFIG>" "$ENV{CudaToolkitLibDir}"
        CONDITION $<CONFIG:Release> "${CMAKE_SOURCE_DIR}/$<CONFIG>" "$ENV{TBB_ROOT}/lib/intel64/vc14" "$ENV{CudaToolkitLibDir}"
    )
endif()

