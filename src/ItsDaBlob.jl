module da_blob

using FileIO
using Images
using LinearAlgebra
using Statistics

"""
A Blob is a mask consisting of blocks of adjacent pixels meeting a threshold.
Only those pixels immediately above, below, left or right are considered to
be adjacent (not diagonals).  Blobs offer easy access to a mask consisting
of pixels in the blob and the perimeter either as a list of steps from
one perimeter point to the next or a list of CartesianIndex with the coordinates
of the perimeter points.
"""
struct Blob
    bounds::CartesianIndices
    mask::BitArray  # Mask of those pixels in the blob
    pstart::CartesianIndex{2} # Start of the perimeter
    psteps::Vector{Tuple{Int,Int}} # Steps around perimeter

    function Blob(bounds::CartesianIndices, mask::BitArray)
        function perimeter(m)  # Computes the perimeter steps
            stps = Tuple{Int,Int}[]
            if prod(size(m)) > 1
                msk(r, c) = checkbounds(Bool, m, r, c) && m[r, c]
                mod8(x) = (x + 7) % 8 + 1 # maintains 1...8
                next(cur, drs) = mod8(drs[findfirst(r -> msk((cur .+ steps[mod8(r)])...), drs)])
                steps = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
                pre = (findfirst(r -> m[r, 1], 1:size(m, 1)), 1)
                @assert msk(pre...)
                ff = findfirst(r -> msk((pre .+ steps[mod8(r)])...), 1:5)
                if isnothing(ff)
                    FileIO.save(File(format"PNG", "dump.png"), mask)
                    @assert !isnothing(ff) "bounds=>$bounds, m[pre]=>$(m[pre...]) m[b]=>$([msk((pre .+ s)...) for s in steps])"
                end
                prevdir = next(pre, 1:5)
                start = pre .+ steps[prevdir]
                @assert msk(start...)
                curr = start
                while true
                    nextdir = next(curr, prevdir-2:prevdir+4)
                    push!(stps, steps[nextdir])
                    curr = curr .+ steps[nextdir]
                    @assert msk(curr...)
                    prevdir = nextdir
                    if curr == start && steps[next(curr, prevdir-2:prevdir+4)] == stps[1]
                        break # end at the start going the same direction
                    end
                end
                return (CartesianIndex{2}(start), stps)
            else
                return (CartesianIndex{2}(1, 1), stps)
            end
        end
        @assert ndims(mask) == 2
        @assert size(bounds) == size(mask)
        ci, stps = perimeter(mask)
        # cip = CartesianIndex([ci.I[i]+bounds.indices[i].start-1 for i in eachindex(ci.I)]...)
        return new(bounds, mask, ci, stps)
    end
end


Base.show(io::IO, b::Blob) = print(io, "Blob[$(b.bounds.indices), area=$(area(b)), perimeter=$(perimeterlength(b))]")


"""
    blob(img::AbstractArray, thresh::Function)::Vector{Blob}

Create a Vector of `Blob`s containing discontiguous regions meeting the threshold function.  The
result is sorted by `Blob` area.
"""
function blob(img::AbstractArray, thresh::Function)::Vector{Blob}
    function extractmask(res, i, minR, maxR, minC, maxC)
        mask = BitArray(undef, maxR - minR + 1, maxC - minC + 1)
        foreach(ci -> mask[ci] = (res[(ci.I .+ (minR - 1, minC - 1))...] == i), CartesianIndices(mask))
        return mask
    end
    res = zeros(UInt32, size(img))
    alias = Vector{Set{eltype(res)}}()
    zz = zero(eltype(res))
    prev, next = zz, zz
    for ci in CartesianIndices(img)
        ci.I[1] == 1 && (prev = zz) == zz  # Reset prev each column
        if thresh(img[ci])
            above = ci[2] > 1 ? res[(ci.I .- (0, 1))...] : zz
            if above ≠ zz
                if (prev ≠ zz) && (above ≠ prev)
                    # Same blob => different indexes
                    ia = findfirst(al -> above in al, alias)
                    ip = findfirst(al -> prev in al, alias)
                    @assert !(isnothing(ia) || isnothing(ip)) "$prev or $above not in $alias"
                    if ia ≠ ip # merge regions
                        union!(alias[ia], alias[ip])
                        deleteat!(alias, ip)
                    end
                end
                prev = above
            elseif prev == zz
                prev = (next += one(next))
                push!(alias, Set{eltype(res)}(prev)) # new region
            end
            @assert prev ≠ zz
            res[ci] = prev
        else
            prev = zz
        end
    end
    # Second pass combine the adjacent indices
    newidx, nblobs = zeros(eltype(res), next), length(alias)
    for i = 1:nblobs, id in alias[i]
        newidx[id] = i
    end
    rects = Dict{eltype(res),NTuple{4,Int}}(i => (100000, -1, 100000, -1) for i = 1:nblobs)
    for ci in CartesianIndices(img)
        if (bidx = res[ci]) ≠ zz # belongs to a blob
            ni = (res[ci] = newidx[bidx])
            rect = rects[ni]
            rects[ni] = (min(ci[1], rect[1]), max(ci[1], rect[2]), min(ci[2], rect[3]), max(ci[2], rect[4]))
        end
    end
    blobs = [
        Blob(CartesianIndices((rect[1]:rect[2], rect[3]:rect[4])), extractmask(res, i, rect...)) for (i, rect) in rects
    ]
    sort!(blobs, lt = (b1, b2) -> area(b1) > area(b2))
    return blobs
end

"""
    Base.CartesianIndices(b::Blob)

Bounds of the blob in the original image's coordinate system
"""
Base.CartesianIndices(b::Blob) = b.bounds

"""
    Base.getindex(b::Blob, ci::CartesianIndex)

Whether a pixel in the original image's coordinate system is in the blob.
"""
Base.getindex(b::Blob, ci::CartesianIndex{2}) = #
    (ci in b.bounds) && b.mask[map(i -> ci.I[i] - b.bounds.indices[i].start + 1, eachindex(ci.I))...]

"""
    perimeter(b::Blob)::Vector{CartesianIndex{2}}

Returns a vector of `CartesianIndex{2}` corresponding to the points around the
perimeter of the blob in the original image's coordinate system.
"""
function perimeter(b::Blob)::Vector{CartesianIndex{2}}
    pts, acc = CartesianIndex{2}[b.pstart], [b.pstart.I...]
    foreach(stp -> push!(pts, CartesianIndex{2}((acc .+= stp)...)), b.psteps[1:end-1])
    return pts
end

function perimeter(b::Blob, img::AxisArray, p::Real=2)
    perim = perimeter(b)
    sum( norm(img, perim[i], perim[i == length(p) ? 1 : i+1], p) for i in eachindex(p) )
end


"""
    perimeterlength(b::Blob)

Compute the length of the blob perimeter in pixels.  Diagonals are √2 and straights are 1.
"""
perimeterlength(b::Blob) = length(b.psteps) > 0 ? mapreduce(st -> sqrt(dot(st, st)), +, b.psteps) : 4.0

"""
    ecd(b::Blob, filled = true)

Computes the equivalent circular diameter (by default computes the ecd(..) based on the area including the area
of any interior holes.)  A = πr² => ecd = 2r = 2√(A/π)
"""
ecd(b::Blob, filled = true) = 2.0 * sqrt((filled ? filledarea(b) : area(b)) / π)

"""
    curvature(b::Blob, n::Int)

Compute an array that measures the angular difference between the pixel n before
and n after then current one for each point on the perimeter of the blob.
Negative indicates convex and positive indicates concave with large positive
values indicating a sharp concave angle.
"""
function curvature(b::Blob, n::Int)
    modn(i) = (i + length(b.psteps) - 1) % length(b.psteps) + 1
    stepsum(itr) = mapreduce(j -> b.psteps[modn(j)], (x, y) -> .+(x, y), itr, init = (0, 0))
    angles = Float64[]
    for i in eachindex(b.psteps)
        sm, sp = -1 .* stepsum(i-1:-1:i-n), stepsum(i:i+n-1)
        den² = dot(sm, sm) * dot(sp, sp)
        ac = den² > 0 ? dot(sm, sp) / sqrt(den²) : 1.0
        @assert (ac < 1.00001) && (ac > -1.00001) "ac=$ac"
        c = sign(sm[1] * sp[2] - sm[2] * sp[1]) / acos(min(1.0, max(-1.0, ac)))
        push!(angles, c)
    end
    return angles
end

"""
    splitblob(b::Blob, p1::CartesianIndex{2}, p2::CartesianIndex{2})

Split a Blob by drawing a line from p1 to p2 (assumed to be on the perimeter
or outside b) and reblobining.
"""
function splitblob(b::Blob, p1::CartesianIndex{2}, p2::CartesianIndex{2})
    mask = copy(b.mask)
    # Draw a line to divide the particles for reblobbing
    drawline(pt->mask[pt...]=false, p1, p2, true)
    res = blob(mask, p -> p)
    # Fix up the hoz and vert
    return map(b2 -> Blob(__offset(b2.bounds, b.bounds), b2.mask), res)
end

"""
    separate(b::Blob, concavity = 0.42, withinterior = true)::Vector{Blob}

Break b into many Blob(s) by looking for concave regions on the perimeter
and joining them with a short(ish) line.
"""
function separate(b::Blob, concavity = 0.42, withinterior = true)::Vector{Blob}
    modn(bb, i) = (i + length(bb.psteps) - 1) % length(bb.psteps) + 1
    # In the concatity vector, find peaks of positive regions
    function findmaxes(c::Vector{Float64}, minwidth::Int, concav::Float64, sym::Int)::Vector{Tuple{Int,Int}}
        first, st, maxi, res = c[1] > 0.0 ? -1 : 0, 0, 0, Vector{Tuple{Int,Int}}()
        for i in eachindex(c)
            if first < 0 # First keeps track of (possible) initial region above zero
                first = c[i] > c[-first] ? -i : (c[i] < 0.0 ? -first : first)
            end
            if st == 0 # In region below zero concavity
                if c[i] > 0.0
                    st, maxi = i, i
                end
            else # In region of positive concavity
                maxi = c[i] > c[maxi] ? i : maxi
                if c[i] < 0.0
                    if i - st >= minwidth && c[maxi] >= concav
                        push!(res, (maxi, sym))
                    end
                    st, maxi = 0, 0 # start new region of below zero concavity
                end
            end
        end
        if first > 0 # Deal with initial region
            lenfirst = (st ≠ 0 ? length(c) - st : 0) + findfirst(i -> c[i] < 0.0, eachindex(c))
            maxi = st ≠ 0 ? (c[maxi] > c[first] ? maxi : first) : first
            if lenfirst >= minwidth && c[maxi] >= concav
                push!(res, (maxi, sym))
            end
        end
        return res
    end
    besti = -1
    if length(b.psteps) > 20
        p, c = perimeter(b), curvature(b, 4)
        maxes = findmaxes(c, 4, concavity, 0) # maximum
        if withinterior
            for (i, ir) in enumerate(interiorregions(b))
                if area(ir) > 20
                    append!(p, perimeter(ir))
                    c = 0.0 .- curvature(ir, 4)
                    append!(maxes, findmaxes(c, 4, concavity, i))
                end
            end
        end
        # Find pairs of concavities to use as splitters
        bestj, bestlen = -1, 100_000_000::Int
        for (i, (ii, symi)) in enumerate(maxes), (ij, symj) in maxes[i+1:end]
            # Only break exterior to exterior, exterior to interior, interior to interior or from one interior to a different interior
            if symi == 0 || symj == 0 || symi != symj
                len = dot((p[ii].I .- p[ij].I),(p[ii].I .- p[ij].I))::Int
                # pick the shortest splitter
                if len < bestlen
                    besti, bestj, bestlen = ii, ij, len
                end
            end
        end
    end
    # There is a split point so split it and recursively separate the splits
    return besti == -1 ? [b] : mapreduce(separate, append!, splitblob(b, p[besti], p[bestj]), init = Blob[])
end


"""
    scorer(b::Blob)

A default function to score a blob as a candidate particle.  Smaller scores are more particle like.
"""
scorer(b::Blob, img::AbstractArray, minarea = 100) = # perimeter/π == ecd for a circle
    (area(b) < minarea ? 100.0 : minarea / area(b)) + perimeterlength(b) / (π * ecd(b, false))

"""
    commonarea(b1::Blob, b2::Blob)

Returns the number of pixels in common between `b1` and `b2`.
"""
commonarea(b1::Blob, b2::Blob) = count(ci -> b1[ci] && b2[ci], intersect(b1, b2))

"""
    multiseparate(img::Array, threshes, score; concavity=0.42, minarea = 10, dump = nothing)

Uses multiple thresholds to attempt to find the best separation of the distinct blobs in the image.  The best blob b
is defined as the one that produces particles that produce smaller `score(b)`.  So a blob will be split if splitting
the blob will produce multiple blobs of lower scores. The default function 'scorer(b::Blob)' looks for more circular
blobs.
"""
function multiseparate(img::AbstractArray, threshes; score = scorer, concavity = 0.42, minarea = 10, dump = nothing)::Vector{Blob}
    function segmenter(th)
        starters = blob(img, p -> p >= th)
        return length(starters) > 0 ? #
               filter(b -> area(b) > minarea, mapreduce(b -> separate(b, concavity), append!, starters)) : #
               Blob[]
    end
    best, cx = segmenter(threshes[1]), 0
    for th in threshes[2:end]
        blobs, newbest = segmenter(th), Blob[]
        # compare the new blobs to the best previous ones
        for bb in best
            # Find which `blobs` make up `bb`
            becomes = filter(b -> commonarea(b, bb) / area(b) > 0.8, blobs)
            # Should we split `bb` into `becomes`
            split = length(becomes) > 1 && mean(map(b->score(b, img), becomes)) < score(bb, img)
            if (!isnothing(dump)) && (length(becomes) > 1)
                open(dump * "[details].txt", "a") do io
                    write(
                        io,
                        "$(basename(dump))[$(cx+=1)][split=$split]: before = $(score(bb)), becomes = $(score.(becomes)), pl = $(perimeterlength(bb)), ecd=$(ecd(bb,false))\n",
                    )
                end
                FileIO.save(
                    File(format"PNG", dump * "[$cx, before, $split].png"),
                    NeXLParticle.colorizedimage([bb], img),
                )
                FileIO.save(
                    File(format"PNG", dump * "[$cx, after, $split].png"),
                    NeXLParticle.colorizedimage(becomes, img),
                )
            end
            if split
                # Use each `becomes` only once
                # deleteat!(blobs, indexin(becomes, blobs))
                append!(newbest, becomes) # split it up...
            else
                push!(newbest, bb)
            end
        end
        best = newbest
    end
    return best
end



"""
   area(b::Blob)

Area of the Blob in pixel count.

area(b::Blob, img::AxisArray)

Area of the Blob in the units associated with `img`.  Assumes that each pixel in the image
is the same size.
"""
area(b::Blob) = Base.count(b.mask)
function area(b::Blob, img::AxisArray)
    pa = prod( abs(last(a)-first(a))/length(a) for a in AxisArrays.axes(img) ) # area of one pixel
    return pa*area(b)
end

"""
    maskedimage(b::Blob, img::AbstractMatrix, mark=missing, markvalue=0.5)

Extract the image data in `img` associate the the Blob `b`.
"""
function maskedimage(b::Blob, img::AbstractMatrix)
    trimmed = copy(img[b.bounds])[:, :]
    foreach(idx -> trimmed[idx] = 0, filter(i -> !b.mask[i], eachindex(trimmed)))
    return trimmed
end

"""
    colorizedimage(bs::Vector{Blob}, img::AbstractArray)
    colorizedimage(chords::Vector{Vector{CartesianIndex{2}}}, img::AbstractArray)

Create a colorized version of img and draw the blob or chords on it.
"""
function colorizedimage(bs::Vector{Blob}, img::AbstractArray)
    off(ci, bs) = [ci.I[i] + bs.bounds.indices[i].start - 1 for i in eachindex(ci.I)]
    clen = min(32,length(bs))
    colors =
        convert.(
            RGB,
            distinguishable_colors(
                clen + 2,
                Color[RGB(253 / 255, 255 / 255, 255 / 255), RGB(0, 0, 0), RGB(0 / 255, 168 / 255, 45 / 255)]
            )[3:end],
        )
    res = RGB.(img)
    for (i, blob) in enumerate(bs)
        c = colors[(i - 1) % clen + 1]
        foreach(ci -> res[ci] = blob[ci] ? 0.5 * c + 0.5 * img[ci] : img[ci], CartesianIndices(blob)) # draw interior
        foreach(ci -> res[off(ci, blob)...] = c, perimeter(blob)) # draw perimeter
        res[off(blob.pstart, blob)...] = RGB(1.0, 0.0, 0.0) # draw start of perimeter...
    end
    return res
end


"""
    intersect(b1::Blob, b2::Blob)

A CartesianIndices with the region in common between b1 and b2.
"""
Base.intersect(b1::Blob, b2::Blob) = CartesianIndices(tuple(map(
    i -> intersect(b1.bounds.indices[i], b2.bounds.indices[i]),
    eachindex(b1.bounds.indices),
)...))

function interiorregions(b::Blob)
    onedge(b1, b2) = any(map(i -> b1[i].start == 1 || b1[i].stop == b2[i].stop - b2[i].start + 1, eachindex(b2)))
    tmp = filter(ib -> !onedge(ib.bounds.indices, b.bounds.indices), blob(b.mask, p -> !p))
    return Blob[Blob(__offset(tb.bounds, b.bounds), tb.mask) for tb in tmp]
end

"""
    filledarea(b::Blob)
    filledarea(b::Blob, img::AxisArray)

Area of the blob plus interior regions.
"""
filledarea(b::Blob) = area(b) + sum(area.(interiorregions(b)))
function filledarea(b::Blob, img::AxisArray)
    pa = prod( abs(last(a)-first(a))/length(a) for a in AxisArrays.axes(img) ) # area of one pixel
    return pa*filledarea(b)
end

function __offset(ci::CartesianIndices{N,R}, base::CartesianIndices{N,R}) where { N, R<:Tuple}
    rs = map(
        i -> ci.indices[i].start+base.indices[i].start-1:ci.indices[i].stop+base.indices[i].start-1,
        eachindex(ci.indices),
    )
    return CartesianIndices{N,R}(tuple(rs...))
end

end # module da_blob
