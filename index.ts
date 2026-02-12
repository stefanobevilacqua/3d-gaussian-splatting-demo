import {vec3, mat3, mat4} from "gl-matrix"

type Model = {
    vertices: vec3[],
    faces: vec3[],
    normals: vec3[]
}
type GaussianSplat = {
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
	    let edge_1_0 = vec3.create()
	    let edge_2_0 = vec3.create()
	    let normal = vec3.create()
	    vec3.sub(edge_1_0, triangle[1]!, triangle[0]!)
	    vec3.sub(edge_2_0, triangle[2]!, triangle[0]!)
	    vec3.cross(normal, edge_1_0, edge_2_0)
	    model.normals.push(normal)
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

function generateGaussianSplats(model: Model, numSplats: number): GaussianSplat[] {
    let areas: number[] = model.normals.map((normal: vec3) => {
	return vec3.length(normal) // No need to multiply by 0.5 here
    })
    let splats: GaussianSplat[] = []
    for(let i = 0; i < numSplats; i++) {
	// TODO (IMPROVEMENT): total area is calculated within randomIndexWithProbability for each splat (can be calculated once)
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
	// let up = vec3.dot(normal, [0, 1, 0]) > 0 ? [0, 1, 0] : [0, -1, 0]
	let up = [0, 1, 0]
	splats.push({
	    position: position,
	    rotation: mat3.fromMat4(mat3.create(), mat4.lookAt(mat4.create(), position, normal, up)),
	    scale: [0.01, 0.01, 0.01],
	    color: [0.9, 0.9, 0.9],
	    opacity: 0.01
	})	
    }
    return splats
}

type RenderContext = {
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    device: GPUDevice,
    splatsBuffer: GPUBuffer,
    pipeline: GPURenderPipeline,
    bindGroup: GPUBindGroup,
    numOfSplats: number,
    mvpArray: Float32Array,
    mvpBuffer: GPUBuffer,
    dt: number,
    et: number,
}

function uploadSplatsToGPU(device: GPUDevice, splats: GaussianSplat[]): GPUBuffer {
    const array: number[] = []
    splats.forEach(splat => {
	array.push(splat.position[0])
	array.push(splat.position[1])
	array.push(splat.position[2])
	array.push(0.0)	// Padding
	array.push(splat.rotation[0 * 3 + 0]!)
	array.push(splat.rotation[0 * 3 + 1]!)
	array.push(splat.rotation[0 * 3 + 2]!)
	array.push(0.0)
	array.push(splat.rotation[1 * 3 + 0]!)
	array.push(splat.rotation[1 * 3 + 1]!)
	array.push(splat.rotation[1 * 3 + 2]!)
	array.push(0.0)
	array.push(splat.rotation[2 * 3 + 0]!)
	array.push(splat.rotation[2 * 3 + 1]!)
	array.push(splat.rotation[2 * 3 + 2]!)
	array.push(0.0)
	array.push(splat.scale[0])
	array.push(splat.scale[1])
	array.push(splat.scale[2])
	array.push(0.0)
	array.push(splat.color[0])
	array.push(splat.color[1])
	array.push(splat.color[2])
	array.push(splat.opacity)
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

    let aspectRatio = ctx.canvas.clientWidth/ctx.canvas.clientHeight
    let view = mat4.create()
    let eyeAngle = ctx.et * 0.001;
    mat4.lookAt(view, [0.3 * Math.sin(eyeAngle), 0.15, 0.3 * Math.cos(eyeAngle)], [0, 0.1, 0], [0, 1, 0])
    let proj = mat4.create()
    mat4.perspectiveZO(proj, Math.PI/3.0, aspectRatio, 0.01, 100.0)
    let mvp = mat4.create()
    mat4.multiply(mvp, proj, view)
    ctx.mvpArray.set(mvp)
    ctx.device.queue.writeBuffer(ctx.mvpBuffer, 0, ctx.mvpArray.buffer)    
    
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
    pass.draw(4, ctx.numOfSplats)
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
    const splats = generateGaussianSplats(model, 100000)
    // const splats = generateGaussianSplats(model, 100)
    
    // tmp
    window.model = model
    window.splats = splats

    const splatsBuffer = uploadSplatsToGPU(device, splats)
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

    let mvpArray = new Float32Array(16)
    let mvpBuffer = device.createBuffer({
	size: mvpArray.byteLength,
	usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    })

    const bindGroup = device.createBindGroup({
	layout: pipeline.getBindGroupLayout(0),
	entries: [{
	    binding: 0,
	    resource: splatsBuffer,
	}, {
	    binding: 1,
	    resource: mvpBuffer,
	}]
    })

    requestAnimationFrame((et: number) => onFrame({
	canvas: canvas,
	context: ctx,
	device: device,
	splatsBuffer: splatsBuffer,
	pipeline: pipeline,
	bindGroup: bindGroup,
	numOfSplats: splats.length,
	mvpArray: mvpArray,
	mvpBuffer: mvpBuffer,
	et: et,
	dt: et
    }))
}

window.addEventListener("DOMContentLoaded", onLoad);
export {}
