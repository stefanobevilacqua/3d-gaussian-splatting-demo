import {vec3, mat3, mat4} from "gl-matrix"

type Model = {
    vertices: vec3[],
    faces: vec3[],
    normals: vec3[]
}
type Gaussian = {
    position: vec3,
    rotation: mat3,
    scale: vec3,
    color: vec3,
    opacity: number,
}

async function getBunnyModel(): Promise<Model> {
    const text = await fetch("bunny.obj").then(response => response.text())
    let model: Model = {vertices: [], faces: [], normals: []}
    text.split(/\r\n|\n|\r/).forEach(line => {
	if(line.startsWith("v")) {
	    model.vertices.push(line.split(/\s+/).slice(1).map(x => parseFloat(x)) as vec3)
	}
	if(line.startsWith("f")) {
	    let face = line.split(/\s+/).slice(1).map(x => parseInt(x) - 1) as vec3
	    model.faces.push(face)
	    let triangle = [
		model.vertices[face[0]],
		model.vertices[face[1]],
		model.vertices[face[2]],
	    ]
	    let normal = vec3.cross(vec3.create(),
				    vec3.sub(vec3.create(), triangle[1]!, triangle[0]!),
				    vec3.sub(vec3.create(), triangle[2]!, triangle[0]!))
	    model.normals.push(vec3.normalize(normal, normal))
	}
    })
    return model
}

function randomIndexWithProbability(weights: number[]) {
    let total = weights.reduce((sum, w) => sum + w);
    let n = Math.random()*total;
    let sum = 0.0;
    for(const [i, w] of weights.entries()) {
	sum += w	
	if(n < sum) return i
    }
    return -1
}

function generateGaussians(model: Model, numGaussians: number): Gaussian[] {
    let areas: number[] = model.normals.map((normal: vec3) => {
	return vec3.length(normal) // No need to multiply by 0.5 here
    })
    let gaussians: Gaussian[] = []
    for(let i = 0; i < numGaussians; i++) {
	// TODO (IMPROVEMENT): total area is calculated within randomIndexWithProbability for each gaussian (can be calculated once)
	const index = randomIndexWithProbability(areas)
	const face = model.faces[index]!
	const triangle = [
	    model.vertices[face[0]],
	    model.vertices[face[1]],
	    model.vertices[face[2]],
	]
	let u = Math.random()
	let v = Math.random()
	if(u + v > 1) {
	    u = 1 - u
	    v = 1 - v
	}
	let position = vec3.scale(vec3.create(), triangle[0]!, 1-u-v)
	vec3.add(position, position, vec3.scale(vec3.create(), triangle[1]!, u))
	vec3.add(position, position, vec3.scale(vec3.create(), triangle[2]!, v))
	let normal = model.normals[index]!
	gaussians.push({
	    position: position,
	    rotation: mat3.fromMat4(mat3.create(),
				    mat4.lookAt(mat4.create(), position, normal, [0, 1, 0])),
	    scale: [0.001, 0.001, 0.001],
	    color: [0.9, 0.9, 0.9],
	    opacity: 0.1
	})	
    }
    return gaussians
}

type RenderContext = {
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    device: GPUDevice,
    gaussiansBuffer: GPUBuffer,
    pipeline: GPURenderPipeline,
    bindGroup: GPUBindGroup,
    numOfGaussians: number,
    cameraViewArray: Float32Array,
    cameraViewBuffer: GPUBuffer,
    aspectRatioArray: Float32Array,
    aspectRatioBuffer: GPUBuffer,
    dt: number,
    et: number,
}

function uploadGaussiansToGPU(device: GPUDevice, gaussians: Gaussian[]): GPUBuffer {
    const array: number[] = []
    gaussians.forEach(g => {
	array.push(g.position[0])
	array.push(g.position[1])
	array.push(g.position[2])
	array.push(0.0)	// Padding
	array.push(g.rotation[0 * 3 + 0]!)
	array.push(g.rotation[0 * 3 + 1]!)
	array.push(g.rotation[0 * 3 + 2]!)
	array.push(0.0)
	array.push(g.rotation[1 * 3 + 0]!)
	array.push(g.rotation[1 * 3 + 1]!)
	array.push(g.rotation[1 * 3 + 2]!)
	array.push(0.0)
	array.push(g.rotation[2 * 3 + 0]!)
	array.push(g.rotation[2 * 3 + 1]!)
	array.push(g.rotation[2 * 3 + 2]!)
	array.push(0.0)
	array.push(g.scale[0])
	array.push(g.scale[1])
	array.push(g.scale[2])
	array.push(0.0)
	array.push(g.color[0])
	array.push(g.color[1])
	array.push(g.color[2])
	array.push(g.opacity)
    })
    const cpuBufferView = new Float32Array(array)
    const buffer = device.createBuffer({
	size: cpuBufferView.byteLength,
	usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    })
    device.queue.writeBuffer(buffer, 0, cpuBufferView.buffer)
    return buffer
}

function onFrame(ctx: RenderContext) {
    ctx.canvas.width = ctx.canvas.clientWidth
    ctx.canvas.height = ctx.canvas.clientHeight

    ctx.aspectRatioArray[0] = ctx.canvas.clientWidth/ctx.canvas.clientHeight
    ctx.device.queue.writeBuffer(ctx.aspectRatioBuffer, 0, ctx.aspectRatioArray.buffer)
    let eyeAngle = ctx.et * 0.001;
    let view = mat4.lookAt(mat4.create(),
			   [0.3 * Math.sin(eyeAngle), 0.15, 0.3 * Math.cos(eyeAngle)],
			   [0, 0.1, 0],
			   [0, 1, 0])
    ctx.cameraViewArray.set(view)
    ctx.device.queue.writeBuffer(ctx.cameraViewBuffer, 0, ctx.cameraViewArray.buffer)    
    
    let cmd = ctx.device.createCommandEncoder()
    let pass = cmd.beginRenderPass({
	colorAttachments: [{
	    loadOp: "clear",
	    storeOp: "store",
	    view: ctx.context.getCurrentTexture().createView(),
	    clearValue: [0.1, 0.2, 0.3, 1.0]
	}]
    })
    pass.setPipeline(ctx.pipeline)
    pass.setBindGroup(0, ctx.bindGroup)
    pass.draw(4, ctx.numOfGaussians)
    pass.end()
    ctx.device.queue.submit([cmd.finish()])
    requestAnimationFrame((et: number) => onFrame({...ctx, et: et, dt: et - ctx.et}))
}

async function onLoad() {
    const adapter = await navigator.gpu?.requestAdapter()
    const device = await adapter?.requestDevice()
    if(!device) {
	console.error("WebGPU is not supported")
	return
    }

    const canvas = document.getElementById("canvas") as HTMLCanvasElement
    const ctx = canvas.getContext("webgpu") as GPUCanvasContext;
    ctx.configure({
	device: device,
	format: navigator.gpu.getPreferredCanvasFormat()
    })

    const model = await getBunnyModel()
    const gaussians = generateGaussians(model, 100000)
    // const gaussians = generateGaussians(model, 1)
    
    const gaussiansBuffer = uploadGaussiansToGPU(device, gaussians)
    const shaderModule = device.createShaderModule({
	code: await fetch("shader.wgsl").then(r => r.text()),
    })
    const pipeline = device.createRenderPipeline({
	layout: "auto",
	vertex: {
	    module: shaderModule,
	    entryPoint: "vs_main",
	},
	primitive: {
	    topology: "triangle-strip"
	},
	fragment: {
	    module: shaderModule,
	    entryPoint: "fs_main",
	    targets: [{
		format: navigator.gpu.getPreferredCanvasFormat(),
		blend: {
		    alpha: {},
		    color: {
			srcFactor: "src-alpha",
			dstFactor: "one-minus-src-alpha",
			operation: "add"
		    }
		}
	    }]
	}
    })

    let cameraViewArray = new Float32Array(16)
    let cameraViewBuffer = device.createBuffer({
	size: cameraViewArray.byteLength,
	usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    })

    let aspectRatioArray = new Float32Array(1)
    let aspectRatioBuffer = device.createBuffer({
	size: aspectRatioArray.byteLength,
	usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    })

    const bindGroup = device.createBindGroup({
	layout: pipeline.getBindGroupLayout(0),
	entries: [{
	    binding: 0,
	    resource: gaussiansBuffer,
	}, {
	    binding: 1,
	    resource: cameraViewBuffer,
	}, {
	    binding: 2,
	    resource: aspectRatioBuffer,
	}]
    })

    requestAnimationFrame((et: number) => onFrame({
	canvas: canvas,
	context: ctx,
	device: device,
	gaussiansBuffer: gaussiansBuffer,
	pipeline: pipeline,
	bindGroup: bindGroup,
	numOfGaussians: gaussians.length,
	cameraViewArray: cameraViewArray,
	cameraViewBuffer: cameraViewBuffer,
	aspectRatioArray: aspectRatioArray,
	aspectRatioBuffer: aspectRatioBuffer,
	et: et,
	dt: et
    }))
}

window.addEventListener("DOMContentLoaded", onLoad);
export {}
